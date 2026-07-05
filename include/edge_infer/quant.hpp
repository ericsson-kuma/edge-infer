#pragma once
// Per-tensor symmetric INT8 quantization primitives.
//
// Scheme: symmetric, per-tensor. A real value x maps to
//     q = clamp(round(x / scale) + zero_point, -128, 127)
// and back via
//     x ~= (q - zero_point) * scale.
// For the symmetric case zero_point == 0 and scale = max(|x|) / 127.
#include <cstddef>
#include <cstdint>

#include "edge_infer/tensor.hpp"

namespace edge {

// Saturating narrow of a 32-bit accumulator to int8 range [-128, 127].
int8_t saturate_i8(int32_t v);

// Calibrate symmetric quant params from data: scale = maxabs / 127, zp = 0.
// An all-zero tensor yields scale = 1.0 (avoids division by zero).
QuantParams compute_symmetric_qparams(const float* data, size_t n);
QuantParams compute_symmetric_qparams(const Tensor& t);

// Quantize / dequantize a single value.
int8_t quantize_value(float x, const QuantParams& qp);
float dequantize_value(int8_t q, const QuantParams& qp);

// Bulk quantize / dequantize over n contiguous elements.
void quantize(const float* in, int8_t* out, size_t n, const QuantParams& qp);
void dequantize(const int8_t* in, float* out, size_t n, const QuantParams& qp);

// Whole-tensor convenience wrappers.
QTensor quantize(const Tensor& t, const QuantParams& qp);
QTensor quantize(const Tensor& t);  // calibrate, then quantize
Tensor dequantize(const QTensor& q);

// Requantize an int32 accumulator (output of an int8 GEMM) back to int8 using a
// real-valued multiplier M = (in_scale * w_scale) / out_scale. This float-path
// requantizer is the readable reference; an integer-only (fixed-point
// multiplier + shift) variant is tracked in BACKLOG.md.
int8_t requantize(int32_t acc, float multiplier, int32_t out_zero_point = 0);
void requantize_row(const int32_t* acc, int8_t* out, size_t n, float multiplier,
                    int32_t out_zero_point = 0);

}  // namespace edge
