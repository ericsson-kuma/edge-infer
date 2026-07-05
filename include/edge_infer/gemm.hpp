#pragma once
// General matrix multiply kernels, row-major: C[M x N] = A[M x K] * B[K x N].
//
// Three variants per dtype form an "optimization journey" that the benchmark
// harness reports:
//   naive     - textbook i,j,k dot-product loop (B is column-strided: poor).
//   reordered - i,k,j loop so the inner loop streams B and C contiguously.
//   blocked   - reordered + cache tiling over K,N + an unrolled inner kernel.
//
// INT8 inputs accumulate into int32. Because integer addition is associative
// and the accumulator cannot overflow for K up to ~131072 (|int8*int8| <=
// 16129), all three INT8 variants are numerically identical -- the tests
// assert this exactly. The FP32 variants differ only by floating-point
// summation order and are compared within tolerance.
#include <cstddef>
#include <cstdint>

namespace edge {

// ---- INT8 (int8 x int8 -> int32) ---------------------------------------
void gemm_i8_naive(const int8_t* A, const int8_t* B, int32_t* C, int M, int N,
                   int K);
void gemm_i8_reordered(const int8_t* A, const int8_t* B, int32_t* C, int M,
                       int N, int K);
void gemm_i8_blocked(const int8_t* A, const int8_t* B, int32_t* C, int M, int N,
                     int K);

// Canonical entry point used by the ops layer.
inline void gemm_i8(const int8_t* A, const int8_t* B, int32_t* C, int M, int N,
                    int K) {
  gemm_i8_blocked(A, B, C, M, N, K);
}

// ---- FP32 (reference precision path) -----------------------------------
void gemm_f32_naive(const float* A, const float* B, float* C, int M, int N,
                    int K);
void gemm_f32_reordered(const float* A, const float* B, float* C, int M, int N,
                        int K);
void gemm_f32_blocked(const float* A, const float* B, float* C, int M, int N,
                      int K);

inline void gemm_f32(const float* A, const float* B, float* C, int M, int N,
                     int K) {
  gemm_f32_blocked(A, B, C, M, N, K);
}

}  // namespace edge
