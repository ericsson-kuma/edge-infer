# Unit tests. googletest is fetched and pinned via FetchContent so the core
# runtime library itself stays dependency-free.
include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
  GIT_SHALLOW    TRUE
)
# Do not let googletest's own install/target settings leak into our build.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

set(EDGE_INFER_TESTS
  test_tensor
  test_quant
  test_gemm
  test_ops
)

foreach(t ${EDGE_INFER_TESTS})
  add_executable(${t} ${t}.cpp)
  target_link_libraries(${t} PRIVATE edge_infer GTest::gtest_main)
  if(NOT MSVC)
    target_compile_options(${t} PRIVATE -Wall -Wextra)
  endif()
  gtest_discover_tests(${t})
endforeach()
