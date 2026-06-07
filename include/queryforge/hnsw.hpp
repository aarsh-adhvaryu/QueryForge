#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "queryforge/distance.hpp"
#include "queryforge/types.hpp"

// HnswIndex — full multi-layer Hierarchical Navigable Small World graph.
//
// Built on the same ideas as NswIndex, but with a stack of layers:
//   * Layer 0 holds every vector with dense local edges (like NSW).
//   * Each higher layer holds an exponentially smaller random subset with longer-range edges.
// A search starts at the single top node, greedily descends the sparse upper layers to land
// near the query cheaply, then runs the wide beam search only at layer 0. This makes search
// cost roughly logarithmic in the number of vectors instead of linear.
//
// Build A3 is done in two steps for clarity:
//   Step 1 (this file's initial form): layers + naive "closest-M" neighbor selection.
//   Step 2: swap select_neighbors() for the diversity heuristic and re-measure.

namespace queryforge {

class HnswIndex {
 public:
  // dim            — vector dimensionality.
  // M              — max edges per node on upper layers (layer 0 uses 2*M; denser where it matters).
  // ef_construction— beam width while inserting.
  // metric         — L2 or Cosine.
  // seed           — RNG seed for the random layer assignment (set for reproducible builds).
  HnswIndex(std::size_t dim, std::size_t M, std::size_t ef_construction, Metric metric,
            std::uint32_t seed = 100);

  std::uint32_t add(const float* vec);

  std::vector<Neighbor> search(const float* query, std::size_t k, std::size_t ef,
                               SearchStats* stats = nullptr) const;

  std::size_t size() const noexcept { return num_nodes_; }
  std::size_t dim() const noexcept { return dim_; }
  Metric metric() const noexcept { return metric_; }
  int max_layer() const noexcept { return max_layer_; }

 private:
  const float* vector_at(std::uint32_t id) const noexcept { return &vectors_[id * dim_]; }
  float distance(const float* a, const float* b) const noexcept;

  // Max edges a node may keep at a given layer (layer 0 is twice as dense).
  std::size_t max_M_for(int layer) const noexcept { return layer == 0 ? M0_ : M_; }

  // Roll a random maximum layer for a new node: floor(-ln(U(0,1)) * mL).
  int random_layer();

  // ef=1 greedy walk on one layer: keep hopping to a closer neighbor; return the closest node.
  std::uint32_t greedy_search_layer(const float* query, std::uint32_t entry, int layer,
                                    SearchStats* stats) const;

  // Wide beam search on one layer (the A2 algorithm), returns up to ef nearest (unsorted).
  std::vector<Neighbor> search_layer(const float* query, std::uint32_t entry, int layer,
                                     std::size_t ef, SearchStats* stats) const;

  // Choose which neighbors to keep from candidates, capped at `m`. Step 1 = naive closest-m.
  void select_neighbors(std::vector<Neighbor>& candidates, std::size_t m) const;

  // Neighbor list of `node` at `layer` (node must exist at that layer).
  std::vector<std::uint32_t>& links(std::uint32_t node, int layer) {
    return links_[node][static_cast<std::size_t>(layer)];
  }
  const std::vector<std::uint32_t>& links(std::uint32_t node, int layer) const {
    return links_[node][static_cast<std::size_t>(layer)];
  }

  std::size_t dim_;
  std::size_t M_;
  std::size_t M0_;  // = 2*M_
  std::size_t ef_construction_;
  Metric metric_;
  double mL_;  // 1 / ln(M)

  std::mt19937 rng_;

  std::size_t num_nodes_ = 0;
  std::uint32_t entry_point_ = 0;
  int max_layer_ = 0;

  std::vector<float> vectors_;  // num_nodes_ * dim_, contiguous
  // links_[node][layer] = neighbor ids of `node` at `layer`. A node has layers 0..node_layer.
  // (Layer 0 is the hot path; flattening it into a contiguous array is a logged perf TODO.)
  std::vector<std::vector<std::vector<std::uint32_t>>> links_;
};

}  // namespace queryforge
