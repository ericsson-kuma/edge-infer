#include <gtest/gtest.h>

#include "edge_infer/graph.hpp"
#include "edge_infer/metrics.hpp"
#include "edge_infer/models.hpp"

namespace {

TEST(Graph, LenetF32ShapeAndDeterminism) {
  auto model = edge::make_lenet(2024);
  auto x = edge::make_random_input({1, 28, 28}, 99);
  auto y1 = model.run_f32(x);
  auto y2 = model.run_f32(x);
  ASSERT_EQ(y1.size(), 10);
  EXPECT_EQ(y1.data, y2.data);  // bit-exact determinism
}

TEST(Graph, QuantizedTracksF32EndToEnd) {
  auto model = edge::make_lenet(2024);
  auto calib = edge::make_random_input({1, 28, 28}, 7);
  auto qmodel = model.quantize(calib);

  // Evaluate on multiple inputs distinct from the calibration sample.
  double worst_cos = 1.0;
  float worst_err_steps = 0.0f;
  for (uint32_t seed : {100u, 101u, 102u, 103u}) {
    auto x = edge::make_random_input({1, 28, 28}, seed);
    auto y_ref = model.run_f32(x);
    auto y_q = qmodel.run(x);
    ASSERT_EQ(y_q.size(), y_ref.size());

    double cos = edge::cosine_similarity(y_ref.ptr(), y_q.ptr(),
                                         static_cast<size_t>(y_ref.size()));
    worst_cos = std::min(worst_cos, cos);

    // Express max error in units of the final layer's quant step.
    auto yqp = edge::compute_symmetric_qparams(y_ref);
    float err = edge::max_abs_err(y_ref.ptr(), y_q.ptr(),
                                  static_cast<size_t>(y_ref.size()));
    worst_err_steps = std::max(worst_err_steps, err / yqp.scale);
  }
  // Accumulated quantization noise through 5 quantized layers; these bounds
  // are what the runtime actually achieves (see README accuracy table).
  EXPECT_GT(worst_cos, 0.99);
  EXPECT_LE(worst_err_steps, 8.0f);
}

TEST(Graph, QuantizedRunIsRepeatable) {
  // Buffer reuse in the static memory plan must not leak state across runs.
  auto model = edge::make_lenet(1);
  auto calib = edge::make_random_input({1, 28, 28}, 2);
  auto qmodel = model.quantize(calib);
  auto x = edge::make_random_input({1, 28, 28}, 3);
  auto y1 = qmodel.run(x);
  auto y2 = qmodel.run(x);
  EXPECT_EQ(y1.data, y2.data);
}

TEST(Graph, ScratchPlanIsBoundedAndNonZero) {
  auto model = edge::make_lenet(5);
  auto calib = edge::make_random_input({1, 28, 28}, 6);
  auto qmodel = model.quantize(calib);
  // Largest activation is conv1 output 6x28x28 = 4704 int8 elements (x2
  // ping-pong) + im2col scratch 25*784 = 19600 -> plan stays well under 64 KiB.
  EXPECT_GT(qmodel.scratch_bytes(), 0u);
  EXPECT_LT(qmodel.scratch_bytes(), 64u * 1024u);
}

TEST(Graph, MixedGraphSmallShapes) {
  // A tiny conv->relu->pool->flatten->fc graph with hand-checkable sizes.
  edge::SequentialModel m;
  edge::Conv2dSpec c{1, 6, 6, 2, 3, 3, 1, 0};  // -> 2x4x4
  std::vector<float> W(2 * 1 * 3 * 3, 0.1f);
  m.add_conv2d(c, W, {0.5f, -0.5f});
  m.add_relu();
  m.add_maxpool2d({2, 4, 4, 2, 2, 2});  // -> 2x2x2
  m.add_flatten();                       // -> 8
  std::vector<float> Wfc(3 * 8, 0.25f);
  m.add_fc(8, 3, Wfc, {});

  auto x = edge::make_random_input({1, 6, 6}, 42);
  auto y = m.run_f32(x);
  ASSERT_EQ(y.size(), 3);
  // All FC rows are identical, so all outputs must match.
  EXPECT_FLOAT_EQ(y.data[0], y.data[1]);
  EXPECT_FLOAT_EQ(y.data[1], y.data[2]);

  auto qm = m.quantize(x);
  auto yq = qm.run(x);
  ASSERT_EQ(yq.size(), 3);
  EXPECT_GT(edge::cosine_similarity(y.ptr(), yq.ptr(), 3), 0.99);
}

}  // namespace
