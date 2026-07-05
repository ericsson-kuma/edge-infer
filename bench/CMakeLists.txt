# Microbenchmark driver. Hand-rolled harness (bench_util.hpp): warmup,
# auto-sized rep count, median +- MAD. No external benchmark dependency.
add_executable(edge_bench
  bench_main.cpp
)
target_link_libraries(edge_bench PRIVATE edge_infer)
target_include_directories(edge_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(NOT MSVC)
  target_compile_options(edge_bench PRIVATE -Wall -Wextra)
endif()
