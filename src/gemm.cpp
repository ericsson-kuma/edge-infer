#include "edge_infer/gemm.hpp"

#include <algorithm>
#include <cstring>

namespace edge {

// Cache-blocking tile sizes. A KC x NC int8 panel of B is ~64 KiB at 256x256,
// which comfortably fits an L2 cache while a full C row stays warm.
namespace {
constexpr int kKC = 256;
constexpr int kNC = 256;
}  // namespace

// ===================== INT8 =====================

void gemm_i8_naive(const int8_t* A, const int8_t* B, int32_t* C, int M, int N,
                   int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int32_t acc = 0;
      for (int k = 0; k < K; ++k) {
        acc += static_cast<int32_t>(A[i * K + k]) *
               static_cast<int32_t>(B[k * N + j]);
      }
      C[i * N + j] = acc;
    }
  }
}

void gemm_i8_reordered(const int8_t* A, const int8_t* B, int32_t* C, int M,
                       int N, int K) {
  std::memset(C, 0, sizeof(int32_t) * static_cast<size_t>(M) * N);
  for (int i = 0; i < M; ++i) {
    int32_t* Crow = C + static_cast<size_t>(i) * N;
    for (int k = 0; k < K; ++k) {
      const int32_t a = static_cast<int32_t>(A[i * K + k]);
      const int8_t* Brow = B + static_cast<size_t>(k) * N;
      for (int j = 0; j < N; ++j) {
        Crow[j] += a * static_cast<int32_t>(Brow[j]);
      }
    }
  }
}

void gemm_i8_blocked(const int8_t* A, const int8_t* B, int32_t* C, int M, int N,
                     int K) {
  std::memset(C, 0, sizeof(int32_t) * static_cast<size_t>(M) * N);
  for (int kc = 0; kc < K; kc += kKC) {
    const int kmax = std::min(kc + kKC, K);
    for (int nc = 0; nc < N; nc += kNC) {
      const int nmax = std::min(nc + kNC, N);
      for (int i = 0; i < M; ++i) {
        int32_t* Crow = C + static_cast<size_t>(i) * N;
        const int8_t* Arow = A + static_cast<size_t>(i) * K;
        for (int k = kc; k < kmax; ++k) {
          const int32_t a = static_cast<int32_t>(Arow[k]);
          const int8_t* Brow = B + static_cast<size_t>(k) * N;
          int j = nc;
          for (; j + 8 <= nmax; j += 8) {
            Crow[j + 0] += a * static_cast<int32_t>(Brow[j + 0]);
            Crow[j + 1] += a * static_cast<int32_t>(Brow[j + 1]);
            Crow[j + 2] += a * static_cast<int32_t>(Brow[j + 2]);
            Crow[j + 3] += a * static_cast<int32_t>(Brow[j + 3]);
            Crow[j + 4] += a * static_cast<int32_t>(Brow[j + 4]);
            Crow[j + 5] += a * static_cast<int32_t>(Brow[j + 5]);
            Crow[j + 6] += a * static_cast<int32_t>(Brow[j + 6]);
            Crow[j + 7] += a * static_cast<int32_t>(Brow[j + 7]);
          }
          for (; j < nmax; ++j) {
            Crow[j] += a * static_cast<int32_t>(Brow[j]);
          }
        }
      }
    }
  }
}

// ===================== FP32 =====================

void gemm_f32_naive(const float* A, const float* B, float* C, int M, int N,
                    int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = acc;
    }
  }
}

void gemm_f32_reordered(const float* A, const float* B, float* C, int M, int N,
                        int K) {
  std::memset(C, 0, sizeof(float) * static_cast<size_t>(M) * N);
  for (int i = 0; i < M; ++i) {
    float* Crow = C + static_cast<size_t>(i) * N;
    for (int k = 0; k < K; ++k) {
      const float a = A[i * K + k];
      const float* Brow = B + static_cast<size_t>(k) * N;
      for (int j = 0; j < N; ++j) {
        Crow[j] += a * Brow[j];
      }
    }
  }
}

void gemm_f32_blocked(const float* A, const float* B, float* C, int M, int N,
                      int K) {
  std::memset(C, 0, sizeof(float) * static_cast<size_t>(M) * N);
  for (int kc = 0; kc < K; kc += kKC) {
    const int kmax = std::min(kc + kKC, K);
    for (int nc = 0; nc < N; nc += kNC) {
      const int nmax = std::min(nc + kNC, N);
      for (int i = 0; i < M; ++i) {
        float* Crow = C + static_cast<size_t>(i) * N;
        const float* Arow = A + static_cast<size_t>(i) * K;
        for (int k = kc; k < kmax; ++k) {
          const float a = Arow[k];
          const float* Brow = B + static_cast<size_t>(k) * N;
          int j = nc;
          for (; j + 8 <= nmax; j += 8) {
            Crow[j + 0] += a * Brow[j + 0];
            Crow[j + 1] += a * Brow[j + 1];
            Crow[j + 2] += a * Brow[j + 2];
            Crow[j + 3] += a * Brow[j + 3];
            Crow[j + 4] += a * Brow[j + 4];
            Crow[j + 5] += a * Brow[j + 5];
            Crow[j + 6] += a * Brow[j + 6];
            Crow[j + 7] += a * Brow[j + 7];
          }
          for (; j < nmax; ++j) {
            Crow[j] += a * Brow[j];
          }
        }
      }
    }
  }
}

}  // namespace edge
