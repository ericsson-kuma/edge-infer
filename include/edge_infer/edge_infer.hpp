#pragma once
// Umbrella header for the edge-infer runtime. Include this to pull in the
// public API. Individual headers can also be included directly.
#include "edge_infer/tensor.hpp"

namespace edge {
// Semantic version string of the runtime, e.g. "0.1.0".
const char* version();
}  // namespace edge
