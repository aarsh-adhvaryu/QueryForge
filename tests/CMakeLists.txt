# Unit tests, built on GoogleTest.
#
# We fetch GoogleTest at configure time with FetchContent (CMake downloads + builds
# it for us). This means a fresh clone needs internet on the first `cmake` run, but
# no system-wide install — the build stays reproducible on any machine (Lightning now,
# the local Core Ultra 9 later).
include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
  # TODO: pin URL_HASH SHA256=... once verified, for fully reproducible fetches.
)
# On Windows this keeps the runtime library consistent; harmless elsewhere.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# One test executable; link it to the engine + GoogleTest's main().
add_executable(qf_tests
  smoke_test.cpp
  distance_test.cpp
  nsw_test.cpp
)
target_link_libraries(qf_tests
  PRIVATE
    QueryForge::queryforge
    GTest::gtest_main
)
# nsw_test.cpp reuses the brute-force ground-truth helpers under tools/.
target_include_directories(qf_tests
  PRIVATE ${CMAKE_SOURCE_DIR}/tools
)

# gtest_discover_tests registers each TEST() with CTest, so `ctest` lists/runs them.
include(GoogleTest)
gtest_discover_tests(qf_tests)
