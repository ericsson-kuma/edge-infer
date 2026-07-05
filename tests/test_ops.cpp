#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "edge_infer/metrics.hpp"
#include "edge_infer/ops.hpp"

namespace {

void fill_f32(std::vector<float>& v, uint32_t seed, float lo = -1.0f,
              float hi = 1.0f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (auto& x : v) x = dist(rng);
}

// ---- ReLU ----------------------------------------------------------------

TEST(Relu, F32) {
  std::vector<float> x = {-2.0f, -0.1f, 0.0f, 0.1f, 3.0f};
  std::vector<float> y(x.size());
  edge::relu_f32(x.data(), y.data(), x.size());
  EXPECT_EQ(y, (std::vector<float>{0.0f, 0.0f, 0.0f, 0.1f, 3.0f}));
}

TEST(Relu, I8MatchesF32ThroughQuantDomain) {
  // With zero_point = 0, relu_i8(quantize(x)) == quantize(relu_f32(x)).
  std::vector<float> x(257);
  fill_f32(x, 7, -2.0f, 2.0f);
  auto qp = edge::compute_symmetric_qparams(x.data(), x.size());
  std::vector<int8_t> xq(x.size());
  edge::quantize(x.data(), xq.data(), x.size(), qp);

  std::vector<int8_t> yq(x.size());
  edge::relu_i8(xq.data(), yq.data(), xq.size(), 0);

  std::vector<float> yf(x.size());
  edge::relu_f32(x.data(), yf.data(), x.size());
  std::vector<int8_t> yf_q(x.size());
  edge::quantize(yf.data(), yf_q.data(), yf.size(), qp);

  EXPECT_EQ(yq, yf_q);
}

// ---- MaxPool ---------------------------------------------------------------

TEST(MaxPool, F32KnownCase) {
  // 1 channel, 4x4, 2x2 window stride 2.
  std::vector<float> x = {1, 2, 5, 6,   //
                          3, 4, 7, 8,   //
                          9, 10, 13, 14,  //
                          11, 12, 15, 16};
  edge::Pool2dSpec s{1, 4, 4, 2, 2, 2};
  std::vector<float> y(4);
  edge::maxpool2d_f32(x.data(), y.data(), s);
  EXPECT_EQ(y, (std::vector<float>{4, 8, 12, 16}));
}

TEST(MaxPool, I8ExactVsQuantizedF32) {
  // Max is order-preserving, so pooling commutes with quantization exactly.
  std::vector<float> x(2 * 8 * 8);
  fill_f32(x, 11);
  auto qp = edge::compute_symmetric_qparams(x.data(), x.size());
  std::vector<int8_t> xq(x.size());
  edge::quantize(x.data(), xq.data(), x.size(), qp);

  edge::Pool2dSpec s{2, 8, 8, 2, 2, 2};
  std::vector<int8_t> yq(2 * 4 * 4);
  edge::maxpool2d_i8(xq.data(), yq.data(), s);

  std::vector<float> yf(2 * 4 * 4);
  edge::maxpool2d_f32(x.data(), yf.data(), s);
  std::vector<int8_t> yf_q(yf.size());
  edge::quantize(yf.data(), yf_q.data(), yf.size(), qp);

  EXPECT_EQ(yq, yf_q);
}

// ---- Fully connected -------------------------------------------------------

TEST(Fc, F32KnownCase) {
  // W = [1 2; 3 4], x = [5, 6], b = [0.5, -0.5] -> y = [17.5, 38.5]
  std::vector<float> W = {1, 2, 3, 4}, x = {5, 6}, b = {0.5f, -0.5f};
  std::vector<float> y(2);
  edge::fc_f32(x.data(), W.data(), b.data(), y.data(), 2, 2);
  EXPECT_FLOAT_EQ(y[0], 17.5f);
  EXPECT_FLOAT_EQ(y[1], 38.5f);
}

TEST(Fc, I8TracksF32Reference) {
  const int IN = 128, OUT = 32;
  std::vector<float> x(IN), W(static_cast<size_t>(OUT) * IN), b(OUT);
  fill_f32(x, 21);
  fill_f32(W, 22, -0.5f, 0.5f);
  fill_f32(b, 23, -0.25f, 0.25f);

  std::vector<float> y_ref(OUT);
  edge::fc_f32(x.data(), W.data(), b.data(), y_ref.data(), IN, OUT);

  auto xqp = edge::compute_symmetric_qparams(x.data(), x.size());
  auto wqp = edge::compute_symmetric_qparams(W.data(), W.size());
  auto yqp = edge::compute_symmetric_qparams(y_ref.data(), y_ref.size());
  std::vector<int8_t> xq(x.size()), Wq(W.size());
  edge::quantize(x.data(), xq.data(), x.size(), xqp);
  edge::quantize(W.data(), Wq.data(), W.size(), wqp);
  auto bq = edge::quantize_bias_i32(b.data(), OUT, xqp.scale, wqp.scale);

  std::vector<int8_t> yq(OUT);
  const float mult = xqp.scale * wqp.scale / yqp.scale;
  edge::fc_i8(xq.data(), Wq.data(), bq.data(), yq.data(), IN, OUT, mult);

  std::vector<float> y_deq(OUT);
  edge::dequantize(yq.data(), y_deq.data(), OUT, yqp);

  EXPECT_LE(edge::max_abs_err(y_ref.data(), y_deq.data(), OUT),
            3.0f * yqp.scale);
  EXPECT_GT(edge::cosine_similarity(y_ref.data(), y_deq.data(), OUT), 0.999);
}

// ---- Conv2d ----------------------------------------------------------------

TEST(Conv2d, F32KnownCaseNoPad) {
  // 1x3x3 input, 1 output channel, 2x2 kernel of ones, stride 1, no pad.
  // Windows sums: [1+2+4+5, 2+3+5+6; 4+5+7+8, 5+6+8+9] = [12 16; 24 28].
  std::vector<float> x = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> W = {1, 1, 1, 1};
  edge::Conv2dSpec s{1, 3, 3, 1, 2, 2, 1, 0};
  std::vector<float> y(static_cast<size_t>(s.out_h()) * s.out_w());
  edge::conv2d_f32(x.data(), W.data(), nullptr, y.data(), s);
  EXPECT_EQ(y, (std::vector<float>{12, 16, 24, 28}));
}

TEST(Conv2d, F32KnownCaseWithPad) {
  // 1x2x2 input, 1x1 output channel via 3x3 ones kernel, pad 1:
  // every output = sum of in-bounds neighbourhood.
  std::vector<float> x = {1, 2, 3, 4};
  std::vector<float> W(9, 1.0f);
  edge::Conv2dSpec s{1, 2, 2, 1, 3, 3, 1, 1};
  ASSERT_EQ(s.out_h(), 2);
  std::vector<float> y(4);
  edge::conv2d_f32(x.data(), W.data(), nullptr, y.data(), s);
  EXPECT_EQ(y, (std::vector<float>{10, 10, 10, 10}));
}

TEST(Conv2d, I8TracksF32Reference) {
  edge::Conv2dSpec s{3, 12, 12, 8, 3, 3, 1, 1};
  const size_t xn = static_cast<size_t>(s.in_c) * s.in_h * s.in_w;
  const size_t wn = static_cast<size_t>(s.out_c) * s.in_c * s.kh * s.kw;
  const size_t yn = static_cast<size_t>(s.out_c) * s.out_h() * s.out_w();
  std::vector<float> x(xn), W(wn), b(s.out_c);
  fill_f32(x, 31);
  fill_f32(W, 32, -0.5f, 0.5f);
  fill_f32(b, 33, -0.25f, 0.25f);

  std::vector<float> y_ref(yn);
  edge::conv2d_f32(x.data(), W.data(), b.data(), y_ref.data(), s);

  auto xqp = edge::compute_symmetric_qparams(x.data(), xn);
  auto wqp = edge::compute_symmetric_qparams(W.data(), wn);
  auto yqp = edge::compute_symmetric_qparams(y_ref.data(), yn);
  std::vector<int8_t> xq(xn), Wq(wn);
  edge::quantize(x.data(), xq.data(), xn, xqp);
  edge::quantize(W.data(), Wq.data(), wn, wqp);
  auto bq = edge::quantize_bias_i32(b.data(), s.out_c, xqp.scale, wqp.scale);

  std::vector<int8_t> yq(yn);
  std::vector<int8_t> scratch(static_cast<size_t>(s.in_c) * s.kh * s.kw *
                              s.out_h() * s.out_w());
  const float mult = xqp.scale * wqp.scale / yqp.scale;
  edge::conv2d_i8(xq.data(), Wq.data(), bq.data(), yq.data(), s, mult,
                  scratch.data());

  std::vector<float> y_deq(yn);
  edge::dequantize(yq.data(), y_deq.data(), yn, yqp);

  EXPECT_LE(edge::max_abs_err(y_ref.data(), y_deq.data(), yn),
            3.0f * yqp.scale);
  EXPECT_GT(edge::cosine_similarity(y_ref.data(), y_deq.data(), yn), 0.999);
}

TEST(Im2col, StridedWithPadding) {
  // 1x3x3 input, 2x2 kernel, stride 2, pad 1 -> out 2x2, rows = 4 taps.
  // Input (quantized domain): 1..9.
  std::vector<int8_t> x = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  edge::Conv2dSpec s{1, 3, 3, 1, 2, 2, 2, 1};
  ASSERT_EQ(s.out_h(), 2);
  ASSERT_EQ(s.out_w(), 2);
  std::vector<int8_t> cols(4 * 4);
  edge::im2col_i8(x.data(), cols.data(), s, 0);
  // Tap (ky=0,kx=0): input coords (-1,-1),(-1,1),(1,-1),(1,1) -> 0,0,0,5
  EXPECT_EQ((std::vector<int8_t>(cols.begin(), cols.begin() + 4)),
            (std::vector<int8_t>{0, 0, 0, 5}));
  // Tap (ky=1,kx=1): coords (0,0),(0,2),(2,0),(2,2) -> 1,3,7,9
  EXPECT_EQ((std::vector<int8_t>(cols.begin() + 12, cols.begin() + 16)),
            (std::vector<int8_t>{1, 3, 7, 9}));
}

}  // namespace
