cmake_minimum_required(VERSION 3.16)
project(edge_infer
        VERSION 0.1.0
        DESCRIPTION "Minimal zero-dependency INT8 quantized inference runtime"
        LANGUAGES CXX)

# ---- Build type ---------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  message(STATUS "No build type set; defaulting to Release")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Options ------------------------------------------------------------
option(EDGE_INFER_BUILD_TESTS "Build unit tests" ON)
option(EDGE_INFER_BUILD_BENCH "Build microbenchmarks" ON)
option(EDGE_INFER_NATIVE      "Compile with -march=native (host tuning)" ON)
option(EDGE_INFER_WERROR      "Treat warnings as errors" OFF)

# ---- Core runtime library (zero third-party dependencies) ---------------
add_library(edge_infer STATIC
  src/version.cpp
)
target_include_directories(edge_infer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(edge_infer PUBLIC cxx_std_17)

# Warnings for the library (kept strict; runtime code is the showcase).
if(MSVC)
  target_compile_options(edge_infer PRIVATE /W4)
else()
  target_compile_options(edge_infer PRIVATE -Wall -Wextra)
  if(EDGE_INFER_WERROR)
    target_compile_options(edge_infer PRIVATE -Werror)
  endif()
endif()

# Optimization knobs applied to Release builds only.
include(CheckCXXCompilerFlag)
if(EDGE_INFER_NATIVE)
  check_cxx_compiler_flag("-march=native" EDGE_INFER_HAS_MARCH_NATIVE)
  if(EDGE_INFER_HAS_MARCH_NATIVE)
    target_compile_options(edge_infer PUBLIC $<$<CONFIG:Release>:-march=native>)
  endif()
endif()
check_cxx_compiler_flag("-funroll-loops" EDGE_INFER_HAS_UNROLL)
if(EDGE_INFER_HAS_UNROLL)
  target_compile_options(edge_infer PUBLIC $<$<CONFIG:Release>:-funroll-loops>)
endif()

# ---- Tests --------------------------------------------------------------
if(EDGE_INFER_BUILD_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
  enable_testing()
  add_subdirectory(tests)
endif()

# ---- Benchmarks ---------------------------------------------------------
if(EDGE_INFER_BUILD_BENCH AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench/CMakeLists.txt)
  add_subdirectory(bench)
endif()
