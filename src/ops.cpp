#include "edge_infer/ops.hpp"

#include <algorithm>
#include <cmath>

namespace edge {

// ---- FP32 reference ------------------------------------------------------

void fc_f32(const float* x, const float* W, const float* b, float* y,
            int in_features, int out_features) {
  for (int o = 0; o < out_features; ++o) {
    float acc = b ? b[o] : 0.0f;
    const float* Wrow = W + static_cast<size_t>(o) * in_features;
    for (int i = 0; i < in_features; ++i) acc += Wrow[i] * x[i];
    y[o] = acc;
  }
}

void conv2d_f32(const float* x, const float* W, const float* b, float* y,
                const Conv2dSpec& s) {
  const int OH = s.out_h(), OW = s.out_w();
  for (int oc = 0; oc < s.out_c; ++oc) {
    for (int oy = 0; oy < OH; ++oy) {
      for (int ox = 0; ox < OW; ++ox) {
        float acc = b ? b[oc] : 0.0f;
        for (int ic = 0; ic < s.in_c; ++ic) {
          for (int ky = 0; ky < s.kh; ++ky) {
            const int iy = oy * s.stride + ky - s.pad;
            if (iy < 0 || iy >= s.in_h) continue;
            for (int kx = 0; kx < s.kw; ++kx) {
              const int ix = ox * s.stride + kx - s.pad;
              if (ix < 0 || ix >= s.in_w) continue;
              const float xv =
                  x[(static_cast<size_t>(ic) * s.in_h + iy) * s.in_w + ix];
              const float wv =
                  W[((static_cast<size_t>(oc) * s.in_c + ic) * s.kh + ky) *
                        s.kw +
                    kx];
              acc += xv * wv;
            }
          }
        }
        y[(static_cast<size_t>(oc) * OH + oy) * OW + ox] = acc;
      }
    }
  }
}

void relu_f32(const float* x, float* y, size_t n) {
  for (size_t i = 0; i < n; ++i) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

void maxpool2d_f32(const float* x, float* y, const Pool2dSpec& s) {
  const int OH = s.out_h(), OW = s.out_w();
  for (int c = 0; c < s.c; ++c) {
    const float* xc = x + static_cast<size_t>(c) * s.in_h * s.in_w;
    float* yc = y + static_cast<size_t>(c) * OH * OW;
    for (int oy = 0; oy < OH; ++oy) {
      for (int ox = 0; ox < OW; ++ox) {
        float m = xc[(oy * s.stride) * s.in_w + (ox * s.stride)];
        for (int ky = 0; ky < s.kh; ++ky) {
          for (int kx = 0; kx < s.kw; ++kx) {
            m = std::max(m, xc[(oy * s.stride + ky) * s.in_w +
                               (ox * s.stride + kx)]);
          }
        }
        yc[oy * OW + ox] = m;
      }
    }
  }
}

// ---- INT8 quantized ------------------------------------------------------

void fc_i8(const int8_t* x, const int8_t* W, const int32_t* b, int8_t* y,
           int in_features, int out_features, float multiplier) {
  for (int o = 0; o < out_features; ++o) {
    int32_t acc = b ? b[o] : 0;
    const int8_t* Wrow = W + static_cast<size_t>(o) * in_features;
    for (int i = 0; i < in_features; ++i) {
      acc += static_cast<int32_t>(Wrow[i]) * static_cast<int32_t>(x[i]);
    }
    y[o] = requantize(acc, multiplier);
  }
}

void im2col_i8(const int8_t* x, int8_t* cols, const Conv2dSpec& s,
               int8_t pad_value) {
  const int OH = s.out_h(), OW = s.out_w();
  const int ON = OH * OW;
  // Row r = (ic, ky, kx); column n = (oy, ox).
  for (int ic = 0; ic < s.in_c; ++ic) {
    for (int ky = 0; ky < s.kh; ++ky) {
      for (int kx = 0; kx < s.kw; ++kx) {
        int8_t* crow =
            cols + (static_cast<size_t>(ic) * s.kh * s.kw + ky * s.kw + kx) *
                       ON;
        for (int oy = 0; oy < OH; ++oy) {
          const int iy = oy * s.stride + ky - s.pad;
          for (int ox = 0; ox < OW; ++ox) {
            const int ix = ox * s.stride + kx - s.pad;
            const bool inside =
                (iy >= 0 && iy < s.in_h && ix >= 0 && ix < s.in_w);
            crow[oy * OW + ox] =
                inside
                    ? x[(static_cast<size_t>(ic) * s.in_h + iy) * s.in_w + ix]
                    : pad_value;
          }
        }
      }
    }
  }
}

void conv2d_i8(const int8_t* x, const int8_t* W, const int32_t* b, int8_t* y,
               const Conv2dSpec& s, float multiplier, int8_t* cols_scratch) {
  const int OH = s.out_h(), OW = s.out_w();
  const int ON = OH * OW;                  // GEMM N
  const int K = s.in_c * s.kh * s.kw;      // GEMM K
  im2col_i8(x, cols_scratch, s, 0);
  // W is [out_c, in_c*kh*kw] when viewed flat; C = W * cols -> [out_c, ON].
  std::vector<int32_t> acc(static_cast<size_t>(s.out_c) * ON);
  gemm_i8(W, cols_scratch, acc.data(), s.out_c, ON, K);
  for (int oc = 0; oc < s.out_c; ++oc) {
    const int32_t bias = b ? b[oc] : 0;
    const int32_t* arow = acc.data() + static_cast<size_t>(oc) * ON;
    int8_t* yrow = y + static_cast<size_t>(oc) * ON;
    for (int n = 0; n < ON; ++n) {
      yrow[n] = requantize(arow[n] + bias, multiplier);
    }
  }
}

void relu_i8(const int8_t* x, int8_t* y, size_t n, int32_t zero_point) {
  const int8_t zp = saturate_i8(zero_point);
  for (size_t i = 0; i < n; ++i) y[i] = x[i] > zp ? x[i] : zp;
}

void maxpool2d_i8(const int8_t* x, int8_t* y, const Pool2dSpec& s) {
  const int OH = s.out_h(), OW = s.out_w();
  for (int c = 0; c < s.c; ++c) {
    const int8_t* xc = x + static_cast<size_t>(c) * s.in_h * s.in_w;
    int8_t* yc = y + static_cast<size_t>(c) * OH * OW;
    for (int oy = 0; oy < OH; ++oy) {
      for (int ox = 0; ox < OW; ++ox) {
        int8_t m = xc[(oy * s.stride) * s.in_w + (ox * s.stride)];
        for (int ky = 0; ky < s.kh; ++ky) {
          for (int kx = 0; kx < s.kw; ++kx) {
            m = std::max(m, xc[(oy * s.stride + ky) * s.in_w +
                               (ox * s.stride + kx)]);
          }
        }
        yc[oy * OW + ox] = m;
      }
    }
  }
}

std::vector<int32_t> quantize_bias_i32(const float* b, int n, float x_scale,
                                       float w_scale) {
  std::vector<int32_t> out(static_cast<size_t>(n));
  const double s = static_cast<double>(x_scale) * w_scale;
  for (int i = 0; i < n; ++i) {
    out[i] = static_cast<int32_t>(std::lround(b[i] / s));
  }
  return out;
}

}  // namespace edge
