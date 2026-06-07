# Developer tools (not part of the shipped library).
add_executable(qf_recall
  recall.cpp
)
target_link_libraries(qf_recall
  PRIVATE QueryForge::queryforge
)
target_include_directories(qf_recall
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}  # for bruteforce.hpp
)
