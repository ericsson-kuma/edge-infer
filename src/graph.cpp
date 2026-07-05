#include "edge_infer/graph.hpp"

#include <cassert>
#include <cstring>

#include "edge_infer/quant.hpp"

namespace edge {

// ---- Builder ---------------------------------------------------------------

SequentialModel& SequentialModel::add_conv2d(const Conv2dSpec& spec,
                                             std::vector<float> W,
                                             std::vector<float> b) {
  Layer l;
  l.kind = LayerKind::Conv2d;
  l.conv = spec;
  l.W = std::move(W);
  l.b = std::move(b);
  assert(static_cast<int64_t>(l.W.size()) ==
         int64_t(spec.out_c) * spec.in_c * spec.kh * spec.kw);
  layers_.push_back(std::move(l));
  return *this;
}

SequentialModel& SequentialModel::add_relu() {
  Layer l;
  l.kind = LayerKind::Relu;
  layers_.push_back(std::move(l));
  return *this;
}

SequentialModel& SequentialModel::add_maxpool2d(const Pool2dSpec& spec) {
  Layer l;
  l.kind = LayerKind::MaxPool2d;
  l.pool = spec;
  layers_.push_back(std::move(l));
  return *this;
}

SequentialModel& SequentialModel::add_flatten() {
  Layer l;
  l.kind = LayerKind::Flatten;
  layers_.push_back(std::move(l));
  return *this;
}

SequentialModel& SequentialModel::add_fc(int in_features, int out_features,
                                         std::vector<float> W,
                                         std::vector<float> b) {
  Layer l;
  l.kind = LayerKind::Fc;
  l.fc_in = in_features;
  l.fc_out = out_features;
  l.W = std::move(W);
  l.b = std::move(b);
  assert(static_cast<int64_t>(l.W.size()) ==
         int64_t(out_features) * in_features);
  layers_.push_back(std::move(l));
  return *this;
}

// ---- FP32 reference pass -----------------------------------------------

namespace {
int64_t layer_out_elems(const Layer& l, int64_t in_elems) {
  switch (l.kind) {
    case LayerKind::Conv2d:
      return int64_t(l.conv.out_c) * l.conv.out_h() * l.conv.out_w();
    case LayerKind::MaxPool2d:
      return int64_t(l.pool.c) * l.pool.out_h() * l.pool.out_w();
    case LayerKind::Fc:
      return l.fc_out;
    case LayerKind::Relu:
    case LayerKind::Flatten:
      return in_elems;
  }
  return in_elems;
}
}  // namespace

Tensor SequentialModel::run_f32(const Tensor& x) const {
  std::vector<float> cur = x.data;
  for (const auto& l : layers_) {
    std::vector<float> next(
        static_cast<size_t>(layer_out_elems(l, static_cast<int64_t>(cur.size()))));
    switch (l.kind) {
      case LayerKind::Conv2d:
        conv2d_f32(cur.data(), l.W.data(), l.b.empty() ? nullptr : l.b.data(),
                   next.data(), l.conv);
        break;
      case LayerKind::Relu:
        relu_f32(cur.data(), next.data(), cur.size());
        break;
      case LayerKind::MaxPool2d:
        maxpool2d_f32(cur.data(), next.data(), l.pool);
        break;
      case LayerKind::Flatten:
        next = cur;
        break;
      case LayerKind::Fc:
        assert(static_cast<int>(cur.size()) == l.fc_in);
        fc_f32(cur.data(), l.W.data(), l.b.empty() ? nullptr : l.b.data(),
               next.data(), l.fc_in, l.fc_out);
        break;
    }
    cur = std::move(next);
  }
  Tensor out({static_cast<int>(cur.size())});
  out.data = std::move(cur);
  return out;
}

// ---- Quantization ---------------------------------------------------------

QuantizedModel SequentialModel::quantize(const Tensor& calibration_input) const {
  QuantizedModel qm;
  qm.input_qp_ = compute_symmetric_qparams(calibration_input);

  // Observe activation ranges with an FP32 pass, layer by layer.
  std::vector<float> act = calibration_input.data;
  QuantParams cur_qp = qm.input_qp_;

  int64_t max_act = static_cast<int64_t>(act.size());
  int64_t max_cols = 0;

  for (const auto& l : layers_) {
    std::vector<float> next(
        static_cast<size_t>(layer_out_elems(l, static_cast<int64_t>(act.size()))));
    QuantizedModel::QLayer ql;
    ql.kind = l.kind;

    switch (l.kind) {
      case LayerKind::Conv2d: {
        conv2d_f32(act.data(), l.W.data(), l.b.empty() ? nullptr : l.b.data(),
                   next.data(), l.conv);
        ql.conv = l.conv;
        QuantParams wqp = compute_symmetric_qparams(l.W.data(), l.W.size());
        QuantParams yqp = compute_symmetric_qparams(next.data(), next.size());
        ql.W.resize(l.W.size());
        edge::quantize(l.W.data(), ql.W.data(), l.W.size(), wqp);
        if (!l.b.empty()) {
          ql.b = quantize_bias_i32(l.b.data(), l.conv.out_c, cur_qp.scale,
                                   wqp.scale);
        }
        ql.multiplier = cur_qp.scale * wqp.scale / yqp.scale;
        ql.out_qp = yqp;
        max_cols = std::max(
            max_cols, int64_t(l.conv.in_c) * l.conv.kh * l.conv.kw *
                          l.conv.out_h() * l.conv.out_w());
        break;
      }
      case LayerKind::Fc: {
        fc_f32(act.data(), l.W.data(), l.b.empty() ? nullptr : l.b.data(),
               next.data(), l.fc_in, l.fc_out);
        ql.fc_in = l.fc_in;
        ql.fc_out = l.fc_out;
        QuantParams wqp = compute_symmetric_qparams(l.W.data(), l.W.size());
        QuantParams yqp = compute_symmetric_qparams(next.data(), next.size());
        ql.W.resize(l.W.size());
        edge::quantize(l.W.data(), ql.W.data(), l.W.size(), wqp);
        if (!l.b.empty()) {
          ql.b = quantize_bias_i32(l.b.data(), l.fc_out, cur_qp.scale,
                                   wqp.scale);
        }
        ql.multiplier = cur_qp.scale * wqp.scale / yqp.scale;
        ql.out_qp = yqp;
        break;
      }
      case LayerKind::Relu:
        relu_f32(act.data(), next.data(), act.size());
        ql.out_qp = cur_qp;  // same scale; exact in quantized domain
        break;
      case LayerKind::MaxPool2d:
        maxpool2d_f32(act.data(), next.data(), l.pool);
        ql.pool = l.pool;
        ql.out_qp = cur_qp;
        break;
      case LayerKind::Flatten:
        next = act;
        ql.out_qp = cur_qp;
        break;
    }

    ql.out_elems = static_cast<int64_t>(next.size());
    max_act = std::max(max_act, ql.out_elems);
    qm.layers_.push_back(std::move(ql));
    act = std::move(next);
    cur_qp = qm.layers_.back().out_qp;
  }

  // Static memory plan: two ping-pong activation buffers + im2col scratch.
  qm.act_a_.resize(static_cast<size_t>(max_act));
  qm.act_b_.resize(static_cast<size_t>(max_act));
  qm.im2col_scratch_.resize(static_cast<size_t>(max_cols));
  return qm;
}

// ---- INT8 execution -----------------------------------------------------

size_t QuantizedModel::scratch_bytes() const {
  return act_a_.size() + act_b_.size() + im2col_scratch_.size();
}

Tensor QuantizedModel::run(const Tensor& x) {
  assert(!layers_.empty());
  assert(x.size() <= static_cast<int64_t>(act_a_.size()));

  int8_t* cur = act_a_.data();
  int8_t* nxt = act_b_.data();
  edge::quantize(x.data.data(), cur, x.data.size(), input_qp_);
  int64_t cur_n = x.size();

  for (const auto& l : layers_) {
    switch (l.kind) {
      case LayerKind::Conv2d:
        conv2d_i8(cur, l.W.data(), l.b.empty() ? nullptr : l.b.data(), nxt,
                  l.conv, l.multiplier, im2col_scratch_.data());
        break;
      case LayerKind::Relu:
        relu_i8(cur, nxt, static_cast<size_t>(cur_n), 0);
        break;
      case LayerKind::MaxPool2d:
        maxpool2d_i8(cur, nxt, l.pool);
        break;
      case LayerKind::Flatten:
        std::memcpy(nxt, cur, static_cast<size_t>(cur_n));
        break;
      case LayerKind::Fc:
        fc_i8(cur, l.W.data(), l.b.empty() ? nullptr : l.b.data(), nxt,
              l.fc_in, l.fc_out, l.multiplier);
        break;
    }
    std::swap(cur, nxt);
    cur_n = l.out_elems;
  }

  Tensor out({static_cast<int>(cur_n)});
  edge::dequantize(cur, out.data.data(), static_cast<size_t>(cur_n),
                   layers_.back().out_qp);
  return out;
}

}  // namespace edge
