#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace queryforge {
namespace detail {

// A reusable "visited" set with O(1) clear, used by the graph searches.
//
// The naive approach allocates and zeroes a `std::vector<bool>(N)` on *every* search. During an
// index build that runs once per insert, so build cost becomes O(N^2). Instead we keep one array of
// version tags around and "clear" it by bumping a counter: a node counts as visited only if its tag
// equals the current version. Bumping the version invalidates every old mark in O(1) — no realloc,
// no re-zeroing.
//
// Callers use a `static thread_local VisitedSet` so each thread gets its own instance: concurrent
// reads stay lock-free and race-free, while a single thread reuses its buffer across calls.
class VisitedSet {
 public:
  // Begin a new search over n nodes. Clears in O(1) via the version tag.
  //
  // Growth is GEOMETRIC and preserves existing entries (resize only zero-fills the *new* slots, and
  // old tags are < the current version so they read as unvisited). This is the crux: a naive
  // assign(n, 0) re-zeroes all N entries, and since the index grows by one node per insert that
  // fires almost every call → O(N) per insert → O(N^2) build. Doubling makes reallocation happen
  // O(log N) times total, i.e. amortized O(1) per call.
  void reset(std::size_t n) {
    if (tags_.size() < n) tags_.resize(std::max(n, tags_.size() * 2), 0);
    if (++version_ == 0) {  // counter wrapped all the way around — do a real clear, once.
      std::fill(tags_.begin(), tags_.end(), 0);
      version_ = 1;
    }
  }

  bool test(std::uint32_t i) const { return tags_[i] == version_; }
  void set(std::uint32_t i) { tags_[i] = version_; }

 private:
  std::vector<std::uint32_t> tags_;
  std::uint32_t version_ = 0;
};

}  // namespace detail
}  // namespace queryforge
