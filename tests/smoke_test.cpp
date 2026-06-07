#include <gtest/gtest.h>

#include "queryforge/version.hpp"

// A0 smoke test: proves the toolchain, the library, and the test runner all work
// end-to-end. Real correctness tests (distance accuracy, recall vs brute force)
// arrive with their stages.
TEST(Smoke, VersionIsReported) {
  EXPECT_STREQ(queryforge::version(), "0.1.0");
}
