# Python bindings (built only when QF_BUILD_PYTHON=ON).
include(FetchContent)

FetchContent_Declare(
  pybind11
  URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz
  # TODO: pin URL_HASH once verified.
)
FetchContent_MakeAvailable(pybind11)

# Module target is named *_py to avoid clashing with the C++ library target `queryforge`,
# but its importable name (the .so) is `queryforge` to match PYBIND11_MODULE(queryforge, ...).
pybind11_add_module(queryforge_py bindings.cpp)
set_target_properties(queryforge_py PROPERTIES OUTPUT_NAME queryforge)
target_link_libraries(queryforge_py PRIVATE QueryForge::queryforge)

# Run the Python demo as a CTest case so the bindings stay verified (`ctest`).
if(QF_BUILD_TESTS)
  find_program(QF_PYTHON NAMES python3 python REQUIRED)
  add_test(NAME python_bindings
           COMMAND ${QF_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/example.py)
  set_tests_properties(python_bindings PROPERTIES
    ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:queryforge_py>")

  # End-to-end demo pipeline dry-run (images -> embeddings -> index -> metadata -> search).
  add_test(NAME python_pipeline
           COMMAND ${QF_PYTHON} -m qf_pipeline.dry_run)
  set_tests_properties(python_pipeline PROPERTIES
    ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:queryforge_py>:${CMAKE_CURRENT_SOURCE_DIR}")
endif()
