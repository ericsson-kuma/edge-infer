#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "edge_infer/gemm.hpp"

namespace {

void fill_i8(std::vector<int8_t>& v, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(-128, 127);
  for (auto& x : v) x = static_cast<int8_t>(dist(rng));
}

void fill_f32(std::vector<float>& v, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : v) x = dist(rng);
}

// All INT8 variants accumulate in int32 with no reassociation hazards, so
// they must agree bit-exactly with the naive reference.
void expect_i8_variants_exact(int M, int N, int K, uint32_t seed) {
  std::vector<int8_t> A(static_cast<size_t>(M) * K), B(static_cast<size_t>(K) * N);
  fill_i8(A, seed);
  fill_i8(B, seed + 1);
  std::vector<int32_t> Cn(static_cast<size_t>(M) * N),
      Cr(static_cast<size_t>(M) * N), Cb(static_cast<size_t>(M) * N);
  edge::gemm_i8_naive(A.data(), B.data(), Cn.data(), M, N, K);
  edge::gemm_i8_reordered(A.data(), B.data(), Cr.data(), M, N, K);
  edge::gemm_i8_blocked(A.data(), B.data(), Cb.data(), M, N, K);
  EXPECT_EQ(Cn, Cr) << "reordered mismatch at M=" << M << " N=" << N
                    << " K=" << K;
  EXPECT_EQ(Cn, Cb) << "blocked mismatch at M=" << M << " N=" << N
                    << " K=" << K;
}

TEST(GemmI8, VariantsMatchNaiveExactly) {
  expect_i8_variants_exact(1, 1, 1, 42);
  expect_i8_variants_exact(3, 5, 7, 43);     // odd, smaller than unroll width
  expect_i8_variants_exact(8, 8, 8, 44);
  expect_i8_variants_exact(17, 33, 65, 45);  // non-multiples of 8
  expect_i8_variants_exact(64, 64, 64, 46);
  expect_i8_variants_exact(31, 300, 257, 47);  // crosses kKC/kNC tile borders
}

TEST(GemmI8, KnownSmallCase) {
  // A = [1 2; 3 4], B = [5 6; 7 8] -> C = [19 22; 43 50]
  std::vector<int8_t> A = {1, 2, 3, 4}, B = {5, 6, 7, 8};
  std::vector<int32_t> C(4);
  edge::gemm_i8(A.data(), B.data(), C.data(), 2, 2, 2);
  EXPECT_EQ(C, (std::vector<int32_t>{19, 22, 43, 50}));
}

TEST(GemmI8, WorstCaseAccumulationNoOverflow) {
  // K=512 of (-128 * -128) = 512 * 16384 = 8388608, far below INT32_MAX.
  const int K = 512;
  std::vector<int8_t> A(K, -128), B(K, -128);
  std::vector<int32_t> C(1);
  edge::gemm_i8(A.data(), B.data(), C.data(), 1, 1, K);
  EXPECT_EQ(C[0], K * 16384);
}

// FP32 variants only differ by summation order; compare with a relative
// tolerance scaled to K.
void expect_f32_variants_close(int M, int N, int K, uint32_t seed) {
  std::vector<float> A(static_cast<size_t>(M) * K), B(static_cast<size_t>(K) * N);
  fill_f32(A, seed);
  fill_f32(B, seed + 1);
  std::vector<float> Cn(static_cast<size_t>(M) * N),
      Cr(static_cast<size_t>(M) * N), Cb(static_cast<size_t>(M) * N);
  edge::gemm_f32_naive(A.data(), B.data(), Cn.data(), M, N, K);
  edge::gemm_f32_reordered(A.data(), B.data(), Cr.data(), M, N, K);
  edge::gemm_f32_blocked(A.data(), B.data(), Cb.data(), M, N, K);
  const float tol = 1e-5f * static_cast<float>(K);
  for (size_t i = 0; i < Cn.size(); ++i) {
    EXPECT_NEAR(Cn[i], Cr[i], tol);
    EXPECT_NEAR(Cn[i], Cb[i], tol);
  }
}

TEST(GemmF32, VariantsMatchNaiveWithinTolerance) {
  expect_f32_variants_close(3, 5, 7, 52);
  expect_f32_variants_close(17, 33, 65, 53);
  expect_f32_variants_close(64, 64, 64, 54);
  expect_f32_variants_close(31, 300, 257, 55);
}

TEST(GemmF32, KnownSmallCase) {
  std::vector<float> A = {1, 2, 3, 4}, B = {5, 6, 7, 8};
  std::vector<float> C(4);
  edge::gemm_f32(A.data(), B.data(), C.data(), 2, 2, 2);
  EXPECT_FLOAT_EQ(C[0], 19.0f);
  EXPECT_FLOAT_EQ(C[1], 22.0f);
  EXPECT_FLOAT_EQ(C[2], 43.0f);
  EXPECT_FLOAT_EQ(C[3], 50.0f);
}

}  // namespace
