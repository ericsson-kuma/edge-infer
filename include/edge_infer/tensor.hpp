#pragma once
// Minimal dense tensor types for the edge-infer runtime.
// Row-major storage, no strides, no aliasing: an intentionally small surface
// that mirrors how a resource-constrained inference kernel would see memory.
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace edge {

// Shape is a small list of dimension extents, row-major (last dim contiguous).
using Shape = std::vector<int>;

// Number of scalar elements described by a shape (product of extents).
inline int64_t num_elements(const Shape& s) {
  if (s.empty()) return 0;
  int64_t n = 1;
  for (int d : s) n *= static_cast<int64_t>(d);
  return n;
}

// FP32 tensor: the "golden" precision used for the reference path and for
// calibration of the quantized path.
struct Tensor {
  Shape shape;
  std::vector<float> data;

  Tensor() = default;
  explicit Tensor(Shape s)
      : shape(std::move(s)),
        data(static_cast<size_t>(num_elements(shape)), 0.0f) {}
  Tensor(Shape s, std::vector<float> d)
      : shape(std::move(s)), data(std::move(d)) {}

  int64_t size() const { return static_cast<int64_t>(data.size()); }
  float* ptr() { return data.data(); }
  const float* ptr() const { return data.data(); }
};

// Per-tensor symmetric quantization parameters.
// For symmetric quantization zero_point is 0; the field is kept so the
// requantize path can be extended to asymmetric schemes without an ABI change.
struct QuantParams {
  float scale = 1.0f;
  int32_t zero_point = 0;
};

// INT8 tensor with attached quantization parameters.
struct QTensor {
  Shape shape;
  std::vector<int8_t> data;
  QuantParams qp;

  QTensor() = default;
  QTensor(Shape s, QuantParams q)
      : shape(std::move(s)),
        data(static_cast<size_t>(num_elements(shape)), 0),
        qp(q) {}

  int64_t size() const { return static_cast<int64_t>(data.size()); }
  int8_t* ptr() { return data.data(); }
  const int8_t* ptr() const { return data.data(); }
};

}  // namespace edge
