#include "queryforge/version.hpp"

namespace queryforge {

const char* version() noexcept {
  // Kept in sync with project(VERSION ...) in the top-level CMakeLists.txt.
  return "0.1.0";
}

}  // namespace queryforge
