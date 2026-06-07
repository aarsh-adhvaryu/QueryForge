# The QueryForge engine library.
#
# Right now it contains only version.cpp. As stages land, add their .cpp files to
# the sources list below (distance.cpp, nsw.cpp, hnsw.cpp, persistence.cpp, ...).
add_library(queryforge
  version.cpp
  distance.cpp
  nsw.cpp
  hnsw.cpp
  persistence.cpp
)

# PUBLIC include dir: anything linking to `queryforge` (tests, benchmarks, the
# Python module) automatically gets our headers on its include path.
target_include_directories(queryforge
  PUBLIC ${CMAKE_SOURCE_DIR}/include
)

target_compile_features(queryforge PUBLIC cxx_std_17)

# Position-independent code so the static lib can be linked into the shared Python module.
set_target_properties(queryforge PROPERTIES POSITION_INDEPENDENT_CODE ON)

# A namespaced alias so other targets link as QueryForge::queryforge — this is the
# conventional, self-documenting way to refer to the library.
add_library(QueryForge::queryforge ALIAS queryforge)
