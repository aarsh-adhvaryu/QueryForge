#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "queryforge/distance.hpp"

// NswIndex — a single-layer Navigable Small World graph.
//
// This is the foundation that HNSW (stage A3) extends by stacking several of these graphs into
// layers. Here we have just one layer: every vector is a node, connected to up to `M` near
// neighbors. Searching means walking the graph greedily toward the query.
//
// Lifecycle for now (per the project plan): build the index by add()-ing vectors, then search()
// it. Single-threaded build; concurrent reads are safe once building stops. Dynamic concurrent
// insertion comes later.

namespace queryforge {

// Result of a search: one neighbor with its distance to the query (smaller = closer).
struct Neighbor {
  float distance;
  std::uint32_t id;
};

// Optional diagnostics filled in by a search — lets us report "visited only X of N nodes".
struct SearchStats {
  std::size_t nodes_visited = 0;          // how many graph nodes we expanded
  std::size_t distance_computations = 0;  // how many distance() calls we made
};

class NswIndex {
 public:
  // dim            — vector dimensionality.
  // M              — max edges per node (higher = better recall, more memory).
  // ef_construction— beam width used while inserting (higher = better graph, slower build).
  // metric         — L2 or Cosine.
  NswIndex(std::size_t dim, std::size_t M, std::size_t ef_construction, Metric metric);

  // Insert one vector (copied in). Returns the id assigned (0, 1, 2, ... in insertion order).
  std::uint32_t add(const float* vec);

  // Return the k nearest neighbors to `query`, closest first.
  // ef is the beam width (efSearch): larger = higher recall, slower. ef is clamped to >= k.
  std::vector<Neighbor> search(const float* query, std::size_t k, std::size_t ef,
                               SearchStats* stats = nullptr) const;

  std::size_t size() const noexcept { return num_nodes_; }
  std::size_t dim() const noexcept { return dim_; }
  Metric metric() const noexcept { return metric_; }

 private:
  // Pointer to the stored vector for node id (vectors are kept contiguously for cache locality).
  const float* vector_at(std::uint32_t id) const noexcept { return &vectors_[id * dim_]; }

  // Distance between two stored nodes / a raw query and a node. For Cosine, vectors are stored
  // normalized so this is 1 - dot; for L2 it's squared Euclidean.
  float distance(const float* a, const float* b) const noexcept;

  // Greedy beam search from `entry`, returns up to `ef` nearest nodes to `query` (unsorted).
  std::vector<Neighbor> search_layer(const float* query, std::uint32_t entry, std::size_t ef,
                                     SearchStats* stats) const;

  // Neighbor access in the flat link array.
  std::uint32_t* links_of(std::uint32_t id) noexcept { return &links_[id * M_]; }
  const std::uint32_t* links_of(std::uint32_t id) const noexcept { return &links_[id * M_]; }
  std::uint16_t link_count(std::uint32_t id) const noexcept { return link_count_[id]; }

  // Set node `id`'s neighbor list to the closest <= M of `candidates` (A2 naive selection).
  void set_neighbors(std::uint32_t id, std::vector<Neighbor>& candidates);

  // Add a back-edge a->b. If a is already full, keep the M closest of {existing neighbors, b}.
  void add_link(std::uint32_t a, std::uint32_t b);

  std::size_t dim_;
  std::size_t M_;
  std::size_t ef_construction_;
  Metric metric_;

  std::size_t num_nodes_ = 0;
  std::uint32_t entry_point_ = 0;

  std::vector<float> vectors_;            // num_nodes_ * dim_, contiguous
  std::vector<std::uint32_t> links_;      // num_nodes_ * M_, contiguous (flat adjacency)
  std::vector<std::uint16_t> link_count_; // actual neighbor count per node (<= M_)
};

}  // namespace queryforge
