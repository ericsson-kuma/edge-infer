#pragma once
// Neural-net operators, in two precisions:
//   *_f32  - FP32 reference implementations (the "golden" path).
//   *_i8   - INT8 quantized implementations built on the int8 GEMM with
//            int32 accumulation and a saturating requantize to the output
//            tensor's quant params.
//
// Data layout: single-sample CHW (channels, height, width), row-major.
// Conv weights: [out_c, in_c, kh, kw]; FC weights: [out_features, in_features].
// Zero padding is exact in the symmetric INT8 domain because zero_point == 0.
#include <cstdint>
#include <vector>

#include "edge_infer/gemm.hpp"
#include "edge_infer/quant.hpp"
#include "edge_infer/tensor.hpp"

namespace edge {

// ---- Shape helpers -------------------------------------------------------
struct Conv2dSpec {
  int in_c, in_h, in_w;
  int out_c;
  int kh, kw;
  int stride = 1;
  int pad = 0;

  int out_h() const { return (in_h + 2 * pad - kh) / stride + 1; }
  int out_w() const { return (in_w + 2 * pad - kw) / stride + 1; }
};

struct Pool2dSpec {
  int c, in_h, in_w;
  int kh, kw;
  int stride;

  int out_h() const { return (in_h - kh) / stride + 1; }
  int out_w() const { return (in_w - kw) / stride + 1; }
};

// ---- FP32 reference ------------------------------------------------------
// y[out] = W[out, :] . x + b[out]; b may be nullptr.
void fc_f32(const float* x, const float* W, const float* b, float* y,
            int in_features, int out_features);

// Direct (non-im2col) convolution used as the golden reference.
void conv2d_f32(const float* x, const float* W, const float* b, float* y,
                const Conv2dSpec& s);

void relu_f32(const float* x, float* y, size_t n);

void maxpool2d_f32(const float* x, float* y, const Pool2dSpec& s);

// ---- INT8 quantized ------------------------------------------------------
// Bias is int32, pre-quantized at scale (x_scale * w_scale), zero_point 0.
// The requantize multiplier is (x_scale * w_scale) / y_scale.

void fc_i8(const int8_t* x, const int8_t* W, const int32_t* b, int8_t* y,
           int in_features, int out_features, float multiplier);

// im2col expansion: x (CHW) -> cols [in_c*kh*kw, out_h*out_w]. Padding
// positions are filled with `pad_value` (0 for symmetric INT8, 0.0f for FP32).
void im2col_i8(const int8_t* x, int8_t* cols, const Conv2dSpec& s,
               int8_t pad_value = 0);

// conv2d = im2col + int8 GEMM (int32 accumulate) + bias + requantize.
// `cols_scratch` must hold in_c*kh*kw * out_h*out_w int8 elements; pass an
// external buffer so the graph executor can reuse one allocation.
void conv2d_i8(const int8_t* x, const int8_t* W, const int32_t* b, int8_t* y,
               const Conv2dSpec& s, float multiplier, int8_t* cols_scratch);

// ReLU in the quantized domain: max(q, zero_point); exact vs FP32 for zp=0.
void relu_i8(const int8_t* x, int8_t* y, size_t n, int32_t zero_point = 0);

// Max-pooling is order-preserving, so it is exact in the quantized domain.
void maxpool2d_i8(const int8_t* x, int8_t* y, const Pool2dSpec& s);

// Quantize an FP32 bias vector to int32 at scale (x_scale * w_scale).
std::vector<int32_t> quantize_bias_i32(const float* b, int n, float x_scale,
                                       float w_scale);

}  // namespace edge
