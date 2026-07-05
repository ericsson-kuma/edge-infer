#pragma once
// Sequential graph executor.
//
// SequentialModel holds FP32 layers and runs the reference forward pass.
// SequentialModel::quantize(calibration_input) runs one FP32 pass to observe
// per-activation ranges, quantizes weights/biases per-tensor, and returns a
// QuantizedModel whose run() executes entirely in INT8 between the (quantized)
// input and (dequantized) output.
//
// The quantized executor uses a static memory plan: two ping-pong activation
// buffers plus one im2col scratch buffer, all sized at quantize() time.
// run() performs no heap allocation.
#include <cstdint>
#include <vector>

#include "edge_infer/ops.hpp"
#include "edge_infer/tensor.hpp"

namespace edge {

enum class LayerKind { Conv2d, Relu, MaxPool2d, Flatten, Fc };

struct Layer {
  LayerKind kind;
  Conv2dSpec conv{};   // Conv2d
  Pool2dSpec pool{};   // MaxPool2d
  int fc_in = 0, fc_out = 0;  // Fc
  std::vector<float> W;  // Conv2d [oc,ic,kh,kw] / Fc [out,in]
  std::vector<float> b;  // per-output-channel bias (may be empty)
};

class QuantizedModel;

class SequentialModel {
 public:
  SequentialModel& add_conv2d(const Conv2dSpec& spec, std::vector<float> W,
                              std::vector<float> b);
  SequentialModel& add_relu();
  SequentialModel& add_maxpool2d(const Pool2dSpec& spec);
  SequentialModel& add_flatten();
  SequentialModel& add_fc(int in_features, int out_features,
                          std::vector<float> W, std::vector<float> b);

  // FP32 reference forward pass. `x` must match the first layer's input size.
  Tensor run_f32(const Tensor& x) const;

  // Calibrate activation scales on one representative input and build the
  // INT8 model. Per-tensor symmetric everywhere.
  QuantizedModel quantize(const Tensor& calibration_input) const;

  const std::vector<Layer>& layers() const { return layers_; }

 private:
  std::vector<Layer> layers_;
};

class QuantizedModel {
 public:
  // Quantizes x with the calibrated input scale, runs the INT8 graph, and
  // dequantizes the final activation. No allocation inside.
  Tensor run(const Tensor& x);

  // Peak bytes of the static memory plan (activations + im2col scratch),
  // excluding weights: what an MCU port would need to reserve.
  size_t scratch_bytes() const;

 private:
  friend class SequentialModel;

  struct QLayer {
    LayerKind kind;
    Conv2dSpec conv{};
    Pool2dSpec pool{};
    int fc_in = 0, fc_out = 0;
    std::vector<int8_t> W;
    std::vector<int32_t> b;
    float multiplier = 1.0f;  // (in_scale * w_scale) / out_scale
    QuantParams out_qp;       // activation qparams after this layer
    int64_t out_elems = 0;
  };

  QuantParams input_qp_;
  std::vector<QLayer> layers_;
  // Static memory plan.
  std::vector<int8_t> act_a_, act_b_;
  std::vector<int8_t> im2col_scratch_;
};

}  // namespace edge
