#include "edge_infer/models.hpp"

#include <cmath>
#include <random>

namespace edge {

namespace {
// He-style init: normal(0, sqrt(2/fan_in)). Keeps activations in a sane
// range through deep stacks so quantization sees realistic distributions.
std::vector<float> he_weights(int64_t n, int fan_in, std::mt19937& rng) {
  std::normal_distribution<float> dist(
      0.0f, std::sqrt(2.0f / static_cast<float>(fan_in)));
  std::vector<float> w(static_cast<size_t>(n));
  for (auto& v : w) v = dist(rng);
  return w;
}

std::vector<float> small_bias(int n, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
  std::vector<float> b(static_cast<size_t>(n));
  for (auto& v : b) v = dist(rng);
  return b;
}
}  // namespace

SequentialModel make_lenet(uint32_t seed) {
  std::mt19937 rng(seed);
  SequentialModel m;

  Conv2dSpec c1{1, 28, 28, 6, 5, 5, 1, 2};  // -> 6x28x28
  m.add_conv2d(c1, he_weights(int64_t(6) * 1 * 5 * 5, 1 * 5 * 5, rng),
               small_bias(6, rng));
  m.add_relu();
  m.add_maxpool2d({6, 28, 28, 2, 2, 2});    // -> 6x14x14

  Conv2dSpec c2{6, 14, 14, 16, 5, 5, 1, 0};  // -> 16x10x10
  m.add_conv2d(c2, he_weights(int64_t(16) * 6 * 5 * 5, 6 * 5 * 5, rng),
               small_bias(16, rng));
  m.add_relu();
  m.add_maxpool2d({16, 10, 10, 2, 2, 2});    // -> 16x5x5

  m.add_flatten();                            // -> 400
  m.add_fc(400, 120, he_weights(int64_t(120) * 400, 400, rng),
           small_bias(120, rng));
  m.add_relu();
  m.add_fc(120, 84, he_weights(int64_t(84) * 120, 120, rng),
           small_bias(84, rng));
  m.add_relu();
  m.add_fc(84, 10, he_weights(int64_t(10) * 84, 84, rng),
           small_bias(10, rng));
  return m;
}

Tensor make_random_input(const Shape& shape, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  Tensor t(shape);
  for (auto& v : t.data) v = dist(rng);
  return t;
}

}  // namespace edge
