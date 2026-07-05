#pragma once
// Synthetic model factories used by the end-to-end tests and benchmarks.
// Weights are seeded pseudo-random (He-style fan-in scaling), NOT trained:
// this repo benchmarks runtime correctness/accuracy/performance, not task
// accuracy on a dataset.
#include <cstdint>

#include "edge_infer/graph.hpp"
#include "edge_infer/tensor.hpp"

namespace edge {

// LeNet-5-shaped CNN for 1x28x28 input:
//   conv 6@5x5 pad2 -> relu -> maxpool2
//   conv 16@5x5     -> relu -> maxpool2
//   flatten(400) -> fc120 -> relu -> fc84 -> relu -> fc10
SequentialModel make_lenet(uint32_t seed);

// Uniform random tensor in [-1, 1], seeded.
Tensor make_random_input(const Shape& shape, uint32_t seed);

}  // namespace edge
