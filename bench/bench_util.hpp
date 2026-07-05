#pragma once
// Tiny hand-rolled benchmark harness: no external dependency.
//
// Protocol per configuration:
//   1. warmup runs (untimed) to fault in pages and warm caches/branch pred.
//   2. auto-pick a repetition count so each config takes ~1s wall time
//      (bounded to [5, 21] reps) -- robust on a shared machine.
//   3. report median and MAD (median absolute deviation): both are
//      order statistics, insensitive to the odd preempted outlier rep.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <vector>

namespace bench {

struct Stats {
  double median_ms = 0.0;
  double mad_ms = 0.0;
  int reps = 0;
};

inline double median_of(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// Times fn() over auto-chosen reps; fn must consume its own outputs (e.g.
// accumulate into a checksum) so the optimizer cannot delete the work.
inline Stats run(const std::function<void()>& fn, int warmup = 2,
                 double target_total_ms = 1000.0) {
  using clock = std::chrono::steady_clock;
  for (int i = 0; i < warmup; ++i) fn();

  // One probe rep to size the loop.
  auto t0 = clock::now();
  fn();
  auto t1 = clock::now();
  double once_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  int reps = 5;
  if (once_ms > 0.0) {
    reps = static_cast<int>(target_total_ms / once_ms);
    reps = std::max(5, std::min(21, reps));
  }

  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(reps));
  for (int i = 0; i < reps; ++i) {
    auto a = clock::now();
    fn();
    auto b = clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(b - a).count());
  }

  Stats s;
  s.reps = reps;
  s.median_ms = median_of(samples);
  std::vector<double> dev;
  dev.reserve(samples.size());
  for (double x : samples) dev.push_back(std::abs(x - s.median_ms));
  s.mad_ms = median_of(dev);
  return s;
}

// Global checksum sink; prevents dead-code elimination of benchmark bodies.
extern volatile int64_t sink;

}  // namespace bench
