# QueryForge — top-level build configuration.
#
# Beginner notes (this file is the "recipe" CMake reads to build everything):
#   * cmake_minimum_required / project: declare the toolchain + project name/version.
#   * We target C++17 (works everywhere; our SIMD target is AVX2, see src/).
#   * Options (QF_BUILD_TESTS / QF_BUILD_BENCHMARKS) let you turn parts on/off, e.g.
#       cmake -S . -B build -DQF_BUILD_BENCHMARKS=OFF
#   * Each subdirectory has its own CMakeLists.txt; add_subdirectory() pulls them in.
cmake_minimum_required(VERSION 3.20)

project(QueryForge
        VERSION 0.1.0
        DESCRIPTION "A from-scratch C++ HNSW vector search engine"
        LANGUAGES CXX)

# ---- Language standard -------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Default to an optimized build ------------------------------------------
# Without this, CMake builds with no optimization, which would make our SIMD
# benchmarks meaningless. Release turns on -O2/-O3.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo)
endif()

# Put all binaries in one place so they're easy to find.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---- Options -----------------------------------------------------------------
option(QF_BUILD_TESTS      "Build unit tests (GoogleTest)"       ON)
option(QF_BUILD_BENCHMARKS "Build benchmarks (Google Benchmark)" ON)
option(QF_BUILD_TOOLS      "Build developer tools (recall harness, etc.)" ON)
option(QF_BUILD_PYTHON     "Build the Pybind11 Python module"    OFF)

enable_testing()  # enabled up front so any subdirectory can register CTest cases

# ---- The engine library ------------------------------------------------------
add_subdirectory(src)

# ---- Tools -------------------------------------------------------------------
if(QF_BUILD_TOOLS)
  add_subdirectory(tools)
endif()

# ---- Python bindings ---------------------------------------------------------
if(QF_BUILD_PYTHON)
  add_subdirectory(python)
endif()

# ---- Tests -------------------------------------------------------------------
if(QF_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ---- Benchmarks --------------------------------------------------------------
if(QF_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()

message(STATUS "QueryForge ${PROJECT_VERSION} | build=${CMAKE_BUILD_TYPE} | tests=${QF_BUILD_TESTS} | benchmarks=${QF_BUILD_BENCHMARKS}")
