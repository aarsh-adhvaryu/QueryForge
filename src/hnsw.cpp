#include "queryforge/hnsw.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

#include "visited_set.hpp"

namespace queryforge {
namespace {

void l2_normalize(const float* in, float* out, std::size_t dim) {
  const float norm = std::sqrt(dot(in, in, dim));
  if (norm == 0.0f) {
    std::copy(in, in + dim, out);
    return;
  }
  const float inv = 1.0f / norm;
  for (std::size_t i = 0; i < dim; ++i) out[i] = in[i] * inv;
}

struct CloserFirst {  // min-heap: top() is nearest
  bool operator()(const Neighbor& a, const Neighbor& b) const { return a.distance > b.distance; }
};
struct FartherFirst {  // max-heap: top() is current worst kept result
  bool operator()(const Neighbor& a, const Neighbor& b) const { return a.distance < b.distance; }
};

}  // namespace

HnswIndex::HnswIndex(std::size_t dim, std::size_t M, std::size_t ef_construction, Metric metric,
                     std::uint32_t seed)
    : dim_(dim),
      M_(M),
      M0_(2 * M),
      ef_construction_(ef_construction),
      metric_(metric),
      mL_(1.0 / std::log(static_cast<double>(M))),
      rng_(seed),
      stride0_(2 * M + 1),
      strideU_(M + 1) {}

float HnswIndex::distance(const float* a, const float* b) const noexcept {
  if (metric_ == Metric::Cosine) return 1.0f - dot(a, b, dim_);
  return l2_sqr(a, b, dim_);
}

int HnswIndex::random_layer() {
  std::uniform_real_distribution<double> u(0.0, 1.0);
  double r = u(rng_);
  if (r < std::numeric_limits<double>::min()) r = std::numeric_limits<double>::min();
  return static_cast<int>(std::floor(-std::log(r) * mL_));
}

std::uint32_t HnswIndex::greedy_search_layer(const float* query, std::uint32_t entry, int layer,
                                             SearchStats* stats) const {
  std::uint32_t cur = entry;
  float cur_d = distance(query, vector_at(cur));
  if (stats) stats->distance_computations++;

  bool improved = true;
  while (improved) {
    improved = false;
    const std::uint32_t* b = link_block(cur, layer);
    const std::uint32_t cnt = b[0];
    for (std::uint32_t i = 0; i < cnt; ++i) {
      const std::uint32_t n = b[1 + i];
      const float d = distance(query, vector_at(n));
      if (stats) stats->distance_computations++;
      if (d < cur_d) {
        cur_d = d;
        cur = n;
        improved = true;
      }
    }
  }
  return cur;
}

std::vector<Neighbor> HnswIndex::search_layer(const float* query, std::uint32_t entry, int layer,
                                              std::size_t ef, SearchStats* stats) const {
  // Reusable per-thread visited set: O(1) clear, no per-call allocation (see visited_set.hpp).
  static thread_local detail::VisitedSet visited;
  visited.reset(num_nodes_);
  std::priority_queue<Neighbor, std::vector<Neighbor>, CloserFirst> candidates;
  std::priority_queue<Neighbor, std::vector<Neighbor>, FartherFirst> results;

  const float d0 = distance(query, vector_at(entry));
  if (stats) stats->distance_computations++;
  visited.set(entry);
  candidates.push({d0, entry});
  results.push({d0, entry});

  while (!candidates.empty()) {
    const Neighbor cur = candidates.top();
    if (cur.distance > results.top().distance) break;
    candidates.pop();
    if (stats) stats->nodes_visited++;

    const std::uint32_t* b = link_block(cur.id, layer);
    const std::uint32_t cnt = b[0];
    for (std::uint32_t i = 0; i < cnt; ++i) {
      const std::uint32_t n = b[1 + i];
      // Prefetch the next neighbor's vector (read, low temporal locality) to hide DRAM latency
      // while we compute the distance for this one — the graph access pattern is random.
      if (i + 1 < cnt) __builtin_prefetch(vector_at(b[1 + i + 1]), 0, 1);
      if (visited.test(n)) continue;
      visited.set(n);
      const float d = distance(query, vector_at(n));
      if (stats) stats->distance_computations++;
      if (results.size() < ef || d < results.top().distance) {
        candidates.push({d, n});
        results.push({d, n});
        if (results.size() > ef) results.pop();
      }
    }
  }

  std::vector<Neighbor> out;
  out.reserve(results.size());
  while (!results.empty()) {
    out.push_back(results.top());
    results.pop();
  }
  return out;
}

void HnswIndex::select_neighbors(std::vector<Neighbor>& candidates, std::size_t m) const {
  // Step 2: the diversity heuristic (HNSW paper, Algorithm 4).
  //
  // Consider candidates nearest-first. Accept a candidate c only if it is closer to the base
  // node than it is to every neighbor we've already accepted. Intuition: if c is nearer to an
  // existing pick than to us, then that pick already "covers" c's direction — adding c would be
  // a redundant edge. Rejecting it keeps our M edges pointing in diverse directions, so the
  // graph stays navigable from every side instead of clumping.
  std::sort(candidates.begin(), candidates.end(),
            [](const Neighbor& a, const Neighbor& b) { return a.distance < b.distance; });
  if (candidates.size() <= m) return;

  std::vector<Neighbor> chosen;
  chosen.reserve(m);
  for (const Neighbor& c : candidates) {
    if (chosen.size() >= m) break;
    const float* vc = vector_at(c.id);
    bool diverse = true;
    for (const Neighbor& r : chosen) {
      // distance(c, already-chosen r) vs distance(c, base node = c.distance)
      if (distance(vc, vector_at(r.id)) < c.distance) {
        diverse = false;
        break;
      }
    }
    if (diverse) chosen.push_back(c);
  }
  candidates.swap(chosen);
}

std::uint32_t HnswIndex::add(const float* vec) {
  const std::uint32_t id = static_cast<std::uint32_t>(num_nodes_);
  const int node_layer = random_layer();

  // Store the (normalized, for cosine) vector and allocate empty neighbor lists per layer.
  vectors_.resize((num_nodes_ + 1) * dim_);
  float* dst = &vectors_[id * dim_];
  if (metric_ == Metric::Cosine) {
    l2_normalize(vec, dst, dim_);
  } else {
    std::copy(vec, vec + dim_, dst);
  }
  // Allocate this node's adjacency blocks (counts start at 0). Layer 0 goes in the flat array;
  // upper layers (if any) get one small contiguous block.
  node_layer_.push_back(node_layer);
  links0_.resize((num_nodes_ + 1) * stride0_, 0);
  links_upper_.emplace_back();
  if (node_layer > 0)
    links_upper_.back().assign(static_cast<std::size_t>(node_layer) * strideU_, 0);

  if (num_nodes_ == 0) {  // first node defines the top of the hierarchy
    entry_point_ = id;
    max_layer_ = node_layer;
    num_nodes_ = 1;
    return id;
  }

  std::uint32_t entry = entry_point_;
  const int top = max_layer_;

  // Phase 1: descend through layers above the new node's layer, just to refine the entry point.
  for (int layer = top; layer > node_layer; --layer) {
    entry = greedy_search_layer(dst, entry, layer, nullptr);
  }

  num_nodes_ = id + 1;  // new node now participates in the graph

  // Phase 2: from min(node_layer, top) down to 0, wire the node into each layer.
  for (int layer = std::min(node_layer, top); layer >= 0; --layer) {
    const std::size_t m = max_M_for(layer);
    std::vector<Neighbor> candidates = search_layer(dst, entry, layer, ef_construction_, nullptr);

    // Pick this node's neighbors at this layer and write its contiguous block.
    std::vector<Neighbor> selected = candidates;
    select_neighbors(selected, m);
    std::uint32_t* myb = link_block(id, layer);
    myb[0] = static_cast<std::uint32_t>(selected.size());
    for (std::size_t i = 0; i < selected.size(); ++i) myb[1 + i] = selected[i].id;

    // Add the reverse edges, pruning any neighbor that now exceeds the layer's cap.
    for (const Neighbor& nb : selected) {
      std::uint32_t* nbb = link_block(nb.id, layer);
      const std::uint32_t cnt = nbb[0];
      if (cnt < m) {
        nbb[1 + cnt] = id;  // room: append
        nbb[0] = cnt + 1;
      } else {
        // Full: re-select the m best of {existing neighbors, id} via the diversity heuristic.
        const float* vnb = vector_at(nb.id);
        std::vector<Neighbor> cand;
        cand.reserve(cnt + 1);
        for (std::uint32_t i = 0; i < cnt; ++i)
          cand.push_back({distance(vnb, vector_at(nbb[1 + i])), nbb[1 + i]});
        cand.push_back({distance(vnb, vector_at(id)), id});
        select_neighbors(cand, m);
        nbb[0] = static_cast<std::uint32_t>(cand.size());
        for (std::size_t i = 0; i < cand.size(); ++i) nbb[1 + i] = cand[i].id;
      }
    }

    // Carry the nearest candidate down as the entry point for the next layer.
    if (!candidates.empty()) {
      const auto nearest = std::min_element(
          candidates.begin(), candidates.end(),
          [](const Neighbor& a, const Neighbor& b) { return a.distance < b.distance; });
      entry = nearest->id;
    }
  }

  // If this node is taller than anything seen so far, it becomes the new global entry point.
  if (node_layer > max_layer_) {
    max_layer_ = node_layer;
    entry_point_ = id;
  }
  return id;
}

std::vector<Neighbor> HnswIndex::search(const float* query, std::size_t k, std::size_t ef,
                                        SearchStats* stats) const {
  if (num_nodes_ == 0 || k == 0) return {};
  ef = std::max(ef, k);

  const float* q = query;
  std::vector<float> qbuf;
  if (metric_ == Metric::Cosine) {
    qbuf.resize(dim_);
    l2_normalize(query, qbuf.data(), dim_);
    q = qbuf.data();
  }

  // Descend the sparse upper layers with cheap ef=1 greedy walks to find a great entry point.
  std::uint32_t entry = entry_point_;
  for (int layer = max_layer_; layer >= 1; --layer) {
    entry = greedy_search_layer(q, entry, layer, stats);
  }

  // Wide beam search only at the bottom layer.
  std::vector<Neighbor> res = search_layer(q, entry, 0, ef, stats);
  std::sort(res.begin(), res.end(),
            [](const Neighbor& a, const Neighbor& b) { return a.distance < b.distance; });
  if (res.size() > k) res.resize(k);
  return res;
}

}  // namespace queryforge
