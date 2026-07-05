#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "edge_infer/quant.hpp"

namespace {

TEST(Saturate, Boundaries) {
  EXPECT_EQ(edge::saturate_i8(0), 0);
  EXPECT_EQ(edge::saturate_i8(127), 127);
  EXPECT_EQ(edge::saturate_i8(128), 127);
  EXPECT_EQ(edge::saturate_i8(100000), 127);
  EXPECT_EQ(edge::saturate_i8(-128), -128);
  EXPECT_EQ(edge::saturate_i8(-129), -128);
  EXPECT_EQ(edge::saturate_i8(-100000), -128);
}

TEST(Calibrate, SymmetricScale) {
  std::vector<float> v = {-2.0f, 0.5f, 1.0f, -1.5f};
  auto qp = edge::compute_symmetric_qparams(v.data(), v.size());
  EXPECT_EQ(qp.zero_point, 0);
  EXPECT_FLOAT_EQ(qp.scale, 2.0f / 127.0f);  // maxabs == 2.0
}

TEST(Calibrate, AllZeroFallsBackToUnitScale) {
  std::vector<float> v = {0.0f, 0.0f, 0.0f};
  auto qp = edge::compute_symmetric_qparams(v.data(), v.size());
  EXPECT_FLOAT_EQ(qp.scale, 1.0f);
}

TEST(Quantize, EndpointsMapToRange) {
  edge::QuantParams qp{2.0f / 127.0f, 0};
  EXPECT_EQ(edge::quantize_value(2.0f, qp), 127);
  EXPECT_EQ(edge::quantize_value(-2.0f, qp), -127);
  EXPECT_EQ(edge::quantize_value(0.0f, qp), 0);
  // Values beyond the calibrated range saturate rather than wrap.
  EXPECT_EQ(edge::quantize_value(100.0f, qp), 127);
  EXPECT_EQ(edge::quantize_value(-100.0f, qp), -128);
}

TEST(Quantize, RoundTripErrorBoundedByHalfStep) {
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
  std::vector<float> x(4096);
  for (auto& v : x) v = dist(rng);

  auto qp = edge::compute_symmetric_qparams(x.data(), x.size());
  std::vector<int8_t> q(x.size());
  std::vector<float> xr(x.size());
  edge::quantize(x.data(), q.data(), x.size(), qp);
  edge::dequantize(q.data(), xr.data(), q.size(), qp);

  float maxerr = 0.0f;
  for (size_t i = 0; i < x.size(); ++i)
    maxerr = std::max(maxerr, std::fabs(x[i] - xr[i]));
  // Within the calibrated range the error is at most half a quant step.
  EXPECT_LE(maxerr, 0.5f * qp.scale + 1e-6f);
}

TEST(Quantize, BulkMatchesPerValue) {
  edge::QuantParams qp{0.1f, 0};
  std::vector<float> x = {0.03f, -0.24f, 1.11f, -0.05f, 0.0f};
  std::vector<int8_t> bulk(x.size());
  edge::quantize(x.data(), bulk.data(), x.size(), qp);
  for (size_t i = 0; i < x.size(); ++i)
    EXPECT_EQ(bulk[i], edge::quantize_value(x[i], qp));
}

TEST(Requantize, SaturatesAtBoundaries) {
  // multiplier chosen so the accumulator overshoots the int8 range.
  EXPECT_EQ(edge::requantize(1000, 1.0f), 127);
  EXPECT_EQ(edge::requantize(-1000, 1.0f), -128);
  EXPECT_EQ(edge::requantize(254, 0.5f), 127);   // 254*0.5 = 127
  EXPECT_EQ(edge::requantize(256, 0.5f), 127);   // 128 -> saturate
  EXPECT_EQ(edge::requantize(-256, 0.5f), -128);
  EXPECT_EQ(edge::requantize(0, 12.34f), 0);
}

TEST(Requantize, RoundsToNearest) {
  EXPECT_EQ(edge::requantize(3, 0.5f), 2);   // 1.5 -> 2 (half away from zero)
  EXPECT_EQ(edge::requantize(-3, 0.5f), -2);
  EXPECT_EQ(edge::requantize(10, 0.25f), 3);  // 2.5 -> 3
}

}  // namespace
