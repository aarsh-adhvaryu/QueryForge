#pragma once

#include <cstddef>
#include <cstdint>

// Small types shared by the index implementations (NSW, HNSW).
namespace queryforge {

// One search result: a neighbor id and its distance to the query (smaller = closer).
struct Neighbor {
  float distance;
  std::uint32_t id;
};

// Optional diagnostics a search can fill in — used to report "visited only X of N nodes".
struct SearchStats {
  std::size_t nodes_visited = 0;          // graph nodes expanded
  std::size_t distance_computations = 0;  // distance() calls made
};

}  // namespace queryforge
