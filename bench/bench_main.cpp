// edge-infer microbenchmark driver.
//
// Measures:
//   1. GEMM optimization journey: naive -> reordered -> blocked, INT8 & FP32,
//      across square sizes.
//   2. End-to-end LeNet-shaped CNN: INT8 quantized vs FP32 reference.
//   3. INT8-vs-FP32 accuracy (max abs err, cosine similarity) on that model.
//
// Prints Markdown tables to stdout and writes bench_results.json next to the
// working directory for the docs page generator.
#include <cinttypes>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "bench_util.hpp"
#include "edge_infer/gemm.hpp"
#include "edge_infer/graph.hpp"
#include "edge_infer/metrics.hpp"
#include "edge_infer/models.hpp"
#include "edge_infer/quant.hpp"

namespace bench {
volatile int64_t sink = 0;
}

namespace {

using GemmI8Fn = void (*)(const int8_t*, const int8_t*, int32_t*, int, int,
                          int);
using GemmF32Fn = void (*)(const float*, const float*, float*, int, int, int);

struct GemmResult {
  std::string kernel;
  std::string dtype;
  int size;
  bench::Stats stats;
  double gops;  // billions of multiply-accumulate ops per second
};

double gops_for(int n, double ms) {
  // n^3 MACs; report MAC/s in units of 1e9.
  return (static_cast<double>(n) * n * n) / (ms * 1e6);
}

std::vector<GemmResult> bench_gemm(const std::vector<int>& sizes) {
  std::vector<GemmResult> out;
  std::mt19937 rng(7);
  for (int n : sizes) {
    const size_t nn = static_cast<size_t>(n) * n;
    std::vector<int8_t> Ai(nn), Bi(nn);
    std::vector<float> Af(nn), Bf(nn);
    std::uniform_int_distribution<int> di(-128, 127);
    std::uniform_real_distribution<float> df(-1.0f, 1.0f);
    for (size_t i = 0; i < nn; ++i) {
      Ai[i] = static_cast<int8_t>(di(rng));
      Bi[i] = static_cast<int8_t>(di(rng));
      Af[i] = df(rng);
      Bf[i] = df(rng);
    }
    std::vector<int32_t> Ci(nn);
    std::vector<float> Cf(nn);

    const std::pair<const char*, GemmI8Fn> i8_kernels[] = {
        {"naive", edge::gemm_i8_naive},
        {"reordered", edge::gemm_i8_reordered},
        {"blocked", edge::gemm_i8_blocked},
    };
    for (const auto& kv : i8_kernels) {
      // Not a structured binding: those cannot be lambda-captured in C++17.
      const char* name = kv.first;
      GemmI8Fn fn = kv.second;
      auto s = bench::run([&] {
        fn(Ai.data(), Bi.data(), Ci.data(), n, n, n);
        bench::sink += Ci[nn - 1];
      });
      out.push_back({name, "int8", n, s, gops_for(n, s.median_ms)});
      std::fprintf(stderr, "  gemm %-9s %-5s n=%-4d  %8.2f ms\n", name, "int8",
                   n, s.median_ms);
    }
    const std::pair<const char*, GemmF32Fn> f32_kernels[] = {
        {"naive", edge::gemm_f32_naive},
        {"reordered", edge::gemm_f32_reordered},
        {"blocked", edge::gemm_f32_blocked},
    };
    for (const auto& kv : f32_kernels) {
      const char* name = kv.first;
      GemmF32Fn fn = kv.second;
      auto s = bench::run([&] {
        fn(Af.data(), Bf.data(), Cf.data(), n, n, n);
        bench::sink += static_cast<int64_t>(Cf[nn - 1]);
      });
      out.push_back({name, "fp32", n, s, gops_for(n, s.median_ms)});
      std::fprintf(stderr, "  gemm %-9s %-5s n=%-4d  %8.2f ms\n", name, "fp32",
                   n, s.median_ms);
    }
  }
  return out;
}

const GemmResult& find(const std::vector<GemmResult>& v,
                       const std::string& kernel, const std::string& dtype,
                       int size) {
  for (const auto& r : v) {
    if (r.kernel == kernel && r.dtype == dtype && r.size == size) return r;
  }
  std::fprintf(stderr, "missing result %s/%s/%d\n", kernel.c_str(),
               dtype.c_str(), size);
  std::abort();
}

struct E2eResult {
  bench::Stats fp32, int8;
  float max_abs_err = 0.0f;
  double cosine = 1.0;
  size_t scratch_bytes = 0;
};

E2eResult bench_e2e() {
  auto model = edge::make_lenet(2024);
  auto calib = edge::make_random_input({1, 28, 28}, 7);
  auto qmodel = model.quantize(calib);

  auto x = edge::make_random_input({1, 28, 28}, 100);
  E2eResult r;
  r.scratch_bytes = qmodel.scratch_bytes();

  r.fp32 = bench::run([&] {
    auto y = model.run_f32(x);
    bench::sink += static_cast<int64_t>(y.data[0] * 1000);
  });
  r.int8 = bench::run([&] {
    auto y = qmodel.run(x);
    bench::sink += static_cast<int64_t>(y.data[0] * 1000);
  });
  std::fprintf(stderr, "  e2e lenet fp32 %.3f ms | int8 %.3f ms\n",
               r.fp32.median_ms, r.int8.median_ms);

  // Accuracy across several fresh inputs (worst case reported).
  for (uint32_t seed : {100u, 101u, 102u, 103u, 104u, 105u, 106u, 107u}) {
    auto xi = edge::make_random_input({1, 28, 28}, seed);
    auto yr = model.run_f32(xi);
    auto yq = qmodel.run(xi);
    r.max_abs_err = std::max(
        r.max_abs_err, edge::max_abs_err(yr.ptr(), yq.ptr(),
                                         static_cast<size_t>(yr.size())));
    r.cosine = std::min(
        r.cosine, edge::cosine_similarity(yr.ptr(), yq.ptr(),
                                          static_cast<size_t>(yr.size())));
  }
  return r;
}

void print_markdown(const std::vector<GemmResult>& g,
                    const std::vector<int>& sizes, const E2eResult& e) {
  std::printf("## GEMM optimization journey (median ms, lower is better)\n\n");
  for (const char* dtype : {"int8", "fp32"}) {
    std::printf("**%s**\n\n| kernel |", dtype);
    for (int n : sizes) std::printf(" %dx%d |", n, n);
    std::printf("\n|---|");
    for (size_t i = 0; i < sizes.size(); ++i) std::printf("---|");
    std::printf("\n");
    for (const char* k : {"naive", "reordered", "blocked"}) {
      std::printf("| %s |", k);
      for (int n : sizes) {
        const auto& r = find(g, k, dtype, n);
        std::printf(" %.2f (%.2f GMAC/s) |", r.stats.median_ms, r.gops);
      }
      std::printf("\n");
    }
    std::printf("| **blocked speedup vs naive** |");
    for (int n : sizes) {
      double sp = find(g, "naive", dtype, n).stats.median_ms /
                  find(g, "blocked", dtype, n).stats.median_ms;
      std::printf(" **%.1fx** |", sp);
    }
    std::printf("\n\n");
  }

  std::printf("## INT8 vs FP32 (blocked kernel)\n\n");
  std::printf("| size | fp32 ms | int8 ms | int8 speedup |\n|---|---|---|---|\n");
  for (int n : sizes) {
    const auto& f = find(g, "blocked", "fp32", n);
    const auto& i = find(g, "blocked", "int8", n);
    std::printf("| %dx%d | %.2f | %.2f | %.2fx |\n", n, n, f.stats.median_ms,
                i.stats.median_ms, f.stats.median_ms / i.stats.median_ms);
  }

  std::printf("\n## End-to-end LeNet-shaped CNN (1x28x28)\n\n");
  std::printf(
      "| path | median latency | throughput | notes |\n|---|---|---|---|\n");
  std::printf("| FP32 reference | %.3f ms | %.0f inf/s | direct conv |\n",
              e.fp32.median_ms, 1000.0 / e.fp32.median_ms);
  std::printf(
      "| INT8 quantized | %.3f ms | %.0f inf/s | im2col+GEMM, %.1fx vs FP32, "
      "%zu B scratch |\n",
      e.int8.median_ms, 1000.0 / e.int8.median_ms,
      e.fp32.median_ms / e.int8.median_ms, e.scratch_bytes);
  std::printf("\n## INT8 accuracy vs FP32 golden (worst over 8 inputs)\n\n");
  std::printf("| metric | value |\n|---|---|\n");
  std::printf("| max abs error | %.4f |\n| cosine similarity | %.5f |\n",
              e.max_abs_err, e.cosine);
}

void write_json(const std::vector<GemmResult>& g, const E2eResult& e,
                const char* path) {
  FILE* f = std::fopen(path, "w");
  if (!f) return;
  std::fprintf(f, "{\n  \"gemm\": [\n");
  for (size_t i = 0; i < g.size(); ++i) {
    std::fprintf(f,
                 "    {\"kernel\": \"%s\", \"dtype\": \"%s\", \"size\": %d, "
                 "\"median_ms\": %.4f, \"mad_ms\": %.4f, \"reps\": %d, "
                 "\"gmacs\": %.3f}%s\n",
                 g[i].kernel.c_str(), g[i].dtype.c_str(), g[i].size,
                 g[i].stats.median_ms, g[i].stats.mad_ms, g[i].stats.reps,
                 g[i].gops, i + 1 < g.size() ? "," : "");
  }
  std::fprintf(f,
               "  ],\n  \"e2e\": {\"fp32_ms\": %.4f, \"int8_ms\": %.4f, "
               "\"max_abs_err\": %.5f, \"cosine\": %.6f, \"scratch_bytes\": "
               "%zu}\n}\n",
               e.fp32.median_ms, e.int8.median_ms,
               static_cast<double>(e.max_abs_err), e.cosine, e.scratch_bytes);
  std::fclose(f);
  std::fprintf(stderr, "wrote %s\n", path);
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> sizes = {64, 128, 256, 512};
  if (argc > 1 && std::string(argv[1]) == "--quick") sizes = {64, 128};

  std::fprintf(stderr, "running GEMM suite...\n");
  auto g = bench_gemm(sizes);
  std::fprintf(stderr, "running e2e suite...\n");
  auto e = bench_e2e();

  print_markdown(g, sizes, e);
  write_json(g, e, "bench_results.json");
  // Consume the sink so the whole program has observable output dependence.
  std::fprintf(stderr, "checksum sink: %" PRId64 "\n", bench::sink);
  return 0;
}
