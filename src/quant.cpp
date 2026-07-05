#include "edge_infer/quant.hpp"

#include <cmath>

namespace edge {

int8_t saturate_i8(int32_t v) {
  if (v > 127) return 127;
  if (v < -128) return -128;
  return static_cast<int8_t>(v);
}

QuantParams compute_symmetric_qparams(const float* data, size_t n) {
  float maxabs = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float a = std::fabs(data[i]);
    if (a > maxabs) maxabs = a;
  }
  QuantParams qp;
  qp.zero_point = 0;
  qp.scale = (maxabs > 0.0f) ? (maxabs / 127.0f) : 1.0f;
  return qp;
}

QuantParams compute_symmetric_qparams(const Tensor& t) {
  return compute_symmetric_qparams(t.data.data(), t.data.size());
}

int8_t quantize_value(float x, const QuantParams& qp) {
  // round-half-away-from-zero (std::lround) then saturate.
  int32_t q = static_cast<int32_t>(std::lround(x / qp.scale)) + qp.zero_point;
  return saturate_i8(q);
}

float dequantize_value(int8_t q, const QuantParams& qp) {
  return (static_cast<int32_t>(q) - qp.zero_point) * qp.scale;
}

void quantize(const float* in, int8_t* out, size_t n, const QuantParams& qp) {
  for (size_t i = 0; i < n; ++i) out[i] = quantize_value(in[i], qp);
}

void dequantize(const int8_t* in, float* out, size_t n, const QuantParams& qp) {
  for (size_t i = 0; i < n; ++i) out[i] = dequantize_value(in[i], qp);
}

QTensor quantize(const Tensor& t, const QuantParams& qp) {
  QTensor q(t.shape, qp);
  quantize(t.data.data(), q.data.data(), t.data.size(), qp);
  return q;
}

QTensor quantize(const Tensor& t) {
  return quantize(t, compute_symmetric_qparams(t));
}

Tensor dequantize(const QTensor& q) {
  Tensor t(q.shape);
  dequantize(q.data.data(), t.data.data(), q.data.size(), q.qp);
  return t;
}

int8_t requantize(int32_t acc, float multiplier, int32_t out_zero_point) {
  // Accumulate the scaling in double to keep the reference path exact.
  int32_t v = static_cast<int32_t>(
                  std::lround(static_cast<double>(acc) * multiplier)) +
              out_zero_point;
  return saturate_i8(v);
}

void requantize_row(const int32_t* acc, int8_t* out, size_t n, float multiplier,
                    int32_t out_zero_point) {
  for (size_t i = 0; i < n; ++i)
    out[i] = requantize(acc[i], multiplier, out_zero_point);
}

}  // namespace edge
