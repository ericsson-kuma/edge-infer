#include <gtest/gtest.h>

#include "edge_infer/edge_infer.hpp"

namespace {

TEST(Tensor, ShapeAndSize) {
  edge::Tensor t({2, 3});
  EXPECT_EQ(t.size(), 6);
  ASSERT_EQ(t.shape.size(), 2u);
  EXPECT_EQ(t.shape[0], 2);
  EXPECT_EQ(t.shape[1], 3);
  for (float v : t.data) EXPECT_FLOAT_EQ(v, 0.0f);
}

TEST(Tensor, NumElements) {
  EXPECT_EQ(edge::num_elements({4, 4, 3}), 48);
  EXPECT_EQ(edge::num_elements({}), 0);
  EXPECT_EQ(edge::num_elements({1}), 1);
}

TEST(QTensor, ZeroInitialised) {
  edge::QTensor q({5}, edge::QuantParams{0.25f, 0});
  EXPECT_EQ(q.size(), 5);
  EXPECT_FLOAT_EQ(q.qp.scale, 0.25f);
  EXPECT_EQ(q.qp.zero_point, 0);
  for (int8_t v : q.data) EXPECT_EQ(v, 0);
}

TEST(Version, NonEmpty) {
  ASSERT_NE(edge::version(), nullptr);
  EXPECT_GT(std::string(edge::version()).size(), 0u);
}

}  // namespace
