#pragma once
// Small header-only accuracy metrics used by tests and the benchmark report.
#include <cmath>
#include <cstddef>

namespace edge {

inline float max_abs_err(const float* a, const float* b, size_t n) {
  float m = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float e = std::fabs(a[i] - b[i]);
    if (e > m) m = e;
  }
  return m;
}

inline double cosine_similarity(const float* a, const float* b, size_t n) {
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (size_t i = 0; i < n; ++i) {
    dot += static_cast<double>(a[i]) * b[i];
    na += static_cast<double>(a[i]) * a[i];
    nb += static_cast<double>(b[i]) * b[i];
  }
  if (na == 0.0 || nb == 0.0) return (na == nb) ? 1.0 : 0.0;
  return dot / (std::sqrt(na) * std::sqrt(nb));
}

}  // namespace edge
