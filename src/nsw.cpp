#include "queryforge/nsw.hpp"

#include <algorithm>
#include <cmath>
#include <queue>

namespace queryforge {
namespace {

// Normalize `in` to unit length into `out`. Cosine similarity on unit vectors == dot product,
// so we normalize once at insert time and then treat cosine distance as 1 - dot.
void l2_normalize(const float* in, float* out, std::size_t dim) {
  const float norm = std::sqrt(dot(in, in, dim));
  if (norm == 0.0f) {
    std::copy(in, in + dim, out);
    return;
  }
  const float inv = 1.0f / norm;
  for (std::size_t i = 0; i < dim; ++i) out[i] = in[i] * inv;
}

// Priority-queue orderings. std::priority_queue is a max-heap w.r.t. the comparator's "less".
struct CloserFirst {  // min-heap: top() is the *nearest* node (smallest distance)
  bool operator()(const Neighbor& a, const Neighbor& b) const { return a.distance > b.distance; }
};
struct FartherFirst {  // max-heap: top() is the *worst* result so far (largest distance)
  bool operator()(const Neighbor& a, const Neighbor& b) const { return a.distance < b.distance; }
};

}  // namespace

NswIndex::NswIndex(std::size_t dim, std::size_t M, std::size_t ef_construction, Metric metric)
    : dim_(dim), M_(M), ef_construction_(ef_construction), metric_(metric) {}

float NswIndex::distance(const float* a, const float* b) const noexcept {
  if (metric_ == Metric::Cosine) {
    return 1.0f - dot(a, b, dim_);  // vectors are stored normalized
  }
  return l2_sqr(a, b, dim_);
}

// ---- The core: greedy beam search over one layer ---------------------------------------
std::vector<Neighbor> NswIndex::search_layer(const float* query, std::uint32_t entry,
                                             std::size_t ef, SearchStats* stats) const {
  std::vector<bool> visited(num_nodes_, false);

  std::priority_queue<Neighbor, std::vector<Neighbor>, CloserFirst> candidates;  // frontier
  std::priority_queue<Neighbor, std::vector<Neighbor>, FartherFirst> results;    // best so far

  const float d0 = distance(query, vector_at(entry));
  if (stats) stats->distance_computations++;
  visited[entry] = true;
  candidates.push({d0, entry});
  results.push({d0, entry});

  while (!candidates.empty()) {
    const Neighbor cur = candidates.top();
    // Stop: the closest thing left to explore is already worse than our worst kept result.
    // Because edges only reach nearby points, nothing better is reachable from here.
    if (cur.distance > results.top().distance) break;
    candidates.pop();
    if (stats) stats->nodes_visited++;

    const std::uint32_t* nbrs = links_of(cur.id);
    const std::uint16_t cnt = link_count(cur.id);
    for (std::uint16_t i = 0; i < cnt; ++i) {
      const std::uint32_t n = nbrs[i];
      if (visited[n]) continue;
      visited[n] = true;

      const float d = distance(query, vector_at(n));
      if (stats) stats->distance_computations++;
      if (results.size() < ef || d < results.top().distance) {
        candidates.push({d, n});
        results.push({d, n});
        if (results.size() > ef) results.pop();  // keep only the ef best
      }
    }
  }

  std::vector<Neighbor> out;
  out.reserve(results.size());
  while (!results.empty()) {
    out.push_back(results.top());
    results.pop();
  }
  return out;  // unsorted (heap order); caller sorts if it needs ordering
}

void NswIndex::set_neighbors(std::uint32_t id, std::vector<Neighbor>& candidates) {
  // A2 naive selection: keep the M closest candidates. (A3 swaps in the diversity heuristic.)
  std::sort(candidates.begin(), candidates.end(),
            [](const Neighbor& a, const Neighbor& b) { return a.distance < b.distance; });
  const std::uint16_t keep = static_cast<std::uint16_t>(std::min(candidates.size(), M_));
  std::uint32_t* dst = links_of(id);
  for (std::uint16_t i = 0; i < keep; ++i) dst[i] = candidates[i].id;
  link_count_[id] = keep;
}

void NswIndex::add_link(std::uint32_t a, std::uint32_t b) {
  std::uint32_t* la = links_of(a);
  const std::uint16_t cnt = link_count_[a];
  if (cnt < M_) {  // room: just append
    la[cnt] = b;
    link_count_[a] = static_cast<std::uint16_t>(cnt + 1);
    return;
  }
  // Full: keep the M closest of {current neighbors, b} measured from a.
  const float* va = vector_at(a);
  std::vector<Neighbor> cand;
  cand.reserve(cnt + 1);
  for (std::uint16_t i = 0; i < cnt; ++i) cand.push_back({distance(va, vector_at(la[i])), la[i]});
  cand.push_back({distance(va, vector_at(b)), b});
  std::sort(cand.begin(), cand.end(),
            [](const Neighbor& x, const Neighbor& y) { return x.distance < y.distance; });
  for (std::size_t i = 0; i < M_; ++i) la[i] = cand[i].id;
  link_count_[a] = static_cast<std::uint16_t>(M_);
}

std::uint32_t NswIndex::add(const float* vec) {
  const std::uint32_t id = static_cast<std::uint32_t>(num_nodes_);

  // Store the vector contiguously (normalized for cosine) and reserve its link slots.
  vectors_.resize((num_nodes_ + 1) * dim_);
  float* dst = &vectors_[id * dim_];
  if (metric_ == Metric::Cosine) {
    l2_normalize(vec, dst, dim_);
  } else {
    std::copy(vec, vec + dim_, dst);
  }
  links_.resize((num_nodes_ + 1) * M_, 0);
  link_count_.push_back(0);

  if (num_nodes_ == 0) {  // first node becomes the entry point
    entry_point_ = id;
    num_nodes_ = 1;
    return id;
  }

  // Find nearest existing nodes to the new vector. num_nodes_ still excludes the new node, so
  // search_layer's `visited` is sized for existing nodes only and the new node is unreachable.
  std::vector<Neighbor> candidates = search_layer(dst, entry_point_, ef_construction_, nullptr);
  num_nodes_ = id + 1;  // new node is now part of the graph

  // Link new node -> closest M, then add the reverse edges (with pruning).
  set_neighbors(id, candidates);
  const std::uint32_t* mine = links_of(id);
  const std::uint16_t mine_cnt = link_count(id);
  for (std::uint16_t i = 0; i < mine_cnt; ++i) add_link(mine[i], id);

  return id;
}

std::vector<Neighbor> NswIndex::search(const float* query, std::size_t k, std::size_t ef,
                                       SearchStats* stats) const {
  if (num_nodes_ == 0 || k == 0) return {};
  ef = std::max(ef, k);

  // Normalize the query for cosine so it matches the normalized stored vectors.
  const float* q = query;
  std::vector<float> qbuf;
  if (metric_ == Metric::Cosine) {
    qbuf.resize(dim_);
    l2_normalize(query, qbuf.data(), dim_);
    q = qbuf.data();
  }

  std::vector<Neighbor> res = search_layer(q, entry_point_, ef, stats);
  std::sort(res.begin(), res.end(),
            [](const Neighbor& a, const Neighbor& b) { return a.distance < b.distance; });
  if (res.size() > k) res.resize(k);
  return res;
}

}  // namespace queryforge
