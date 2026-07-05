# edge-infer

[![ci](https://github.com/ericsson-kuma/edge-infer/actions/workflows/ci.yml/badge.svg)](https://github.com/ericsson-kuma/edge-infer/actions/workflows/ci.yml)

A minimal INT8 quantized inference runtime and microbenchmark suite in C++17.
**Zero third-party dependencies in the runtime** — the only external code is
googletest, fetched and pinned for the test suite only.

I built this to demonstrate the systems side of on-device ML: what it takes to
run a small CNN on router/AP-class edge hardware, where you get one CPU core, a
few MB of memory, no vendor BLAS, and no Python. It is a companion piece to my
mesh-steer-ml project (Wi-Fi mesh steering with an INT8 model on the device
side); edge-infer is the "how does that actually execute" layer, written from
scratch.

**What this is:** an engineering benchmark of runtime *correctness, accuracy,
and performance* — quantization round-trips, INT8 kernels validated against an
FP32 golden reference, and honest measurements of an optimization journey.

**What this is not:** ML research. The weights are seeded synthetic (He-init),
not trained; there is no dataset and no claimed task accuracy. Every number
below is reproducible from this repo on commodity hardware.

## Quickstart

```bash
git clone https://github.com/ericsson-kuma/edge-infer.git
cd edge-infer
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build            # 32 tests
./build/bench/edge_bench          # full suite (~2 min), or --quick
```

Requires CMake ≥ 3.16 and any C++17 compiler (CI covers gcc and clang).

## Architecture

```
            +----------------------------------------------------------+
            |                  SequentialModel (FP32)                  |
            |   conv2d_f32 / fc_f32 / relu_f32 / maxpool2d_f32         |
            |   = golden reference + calibration pass                  |
            +-----------------------------+----------------------------+
                                          | quantize(calibration input)
                                          |  - per-tensor symmetric scales
                                          |  - int32 bias @ (s_x * s_w)
                                          |  - requant multipliers
                                          v
            +----------------------------------------------------------+
            |                   QuantizedModel (INT8)                  |
            |                                                          |
            |  input --quantize--> [int8 act] --+                      |
            |                                   |   static memory plan |
            |   conv2d_i8:  im2col -> gemm_i8 --+   (2 ping-pong act   |
            |               (int32 acc) -> +bias    buffers + 1 im2col |
            |               -> requantize (sat)     scratch; no alloc  |
            |   fc_i8:      dot rows -> requantize  inside run())      |
            |   relu_i8 / maxpool2d_i8: native int8, exact             |
            |                                   |                      |
            |  output <--dequantize-- [int8 act]+                      |
            +----------------------------------------------------------+

  gemm_i8 / gemm_f32:   naive  ->  reordered (i,k,j)  ->  blocked + unrolled
                        (the same kernels the benchmark suite measures)
```

Design decisions, briefly:

- **Per-tensor symmetric quantization** (`scale = maxabs/127`, zero-point 0).
  Symmetric means zero padding is *exact* in the quantized domain, and the
  requantize path has no zero-point cross terms. Per-channel is the obvious
  next step (see BACKLOG).
- **INT8 GEMM accumulates in int32**, then a single saturating requantize
  (`multiplier = s_x·s_w / s_y`) narrows back to int8. Worst-case accumulation
  depth before overflow is K ≈ 131k — asserted by test, far beyond these
  shapes.
- **conv2d lowers to im2col + GEMM.** One data-layout transform buys every
  conv the same optimized kernel — the standard mobile-runtime trade (memory
  for locality).
- **Static memory plan.** At quantize() time the executor sizes two ping-pong
  activation buffers and one im2col scratch; `run()` does zero heap
  allocation. The LeNet-shaped model needs **29,008 bytes** of scratch — the
  number an MCU port would need to reserve.
- **ReLU and maxpool run natively in int8** (order-preserving, so exact),
  reusing the input scale.

## Benchmarks

Measured with the hand-rolled harness in `bench/bench_util.hpp`: warmup runs,
auto-sized repetition count (~1 s per config, 5–21 reps), reporting **median**
with MAD as the dispersion measure — order statistics chosen deliberately
because this is a shared machine and the occasional preempted rep should not
pollute results.

| | |
|---|---|
| CPU | Intel Core i7-6700 @ 3.40 GHz (Skylake, 4C/8T, AVX2+FMA) |
| Compiler | g++ 11.4.0 (Ubuntu 22.04) |
| Flags | `-O3 -DNDEBUG -march=native -funroll-loops`, single-threaded |

### GEMM optimization journey (median ms, lower is better)

**int8** (int8×int8 → int32 accumulate)

| kernel | 64×64 | 128×128 | 256×256 | 512×512 |
|---|---|---|---|---|
| naive | 0.09 | 0.73 | 12.67 | 143.69 |
| reordered | 0.03 | 0.19 | 1.51 | 12.04 |
| blocked | 0.03 | 0.20 | 1.55 | 12.85 |
| **blocked speedup vs naive** | **2.9×** | **3.6×** | **8.2×** | **11.2×** |

**fp32**

| kernel | 64×64 | 128×128 | 256×256 | 512×512 |
|---|---|---|---|---|
| naive | 0.17 | 1.91 | 18.05 | 205.00 |
| reordered | 0.02 | 0.15 | 1.66 | 14.78 |
| blocked | 0.02 | 0.14 | 1.53 | 15.00 |
| **blocked speedup vs naive** | **8.7×** | **13.2×** | **11.8×** | **13.7×** |

The journey, in cache terms: **naive** walks B column-wise (one useful element
per cache line); **reordered** (i,k,j) streams B and C rows contiguously, which
the compiler auto-vectorizes — worth ~9–12× alone at large sizes; **blocked**
adds 256×256 K/N tiling and an 8-wide unrolled inner loop, which matters once
the working set falls out of L2 (at 512×512, reordered starts losing steam:
int8 11.14 GMAC/s reordered vs 0.93 naive).

### INT8 vs FP32 (blocked kernel, same machine)

| size | fp32 ms | int8 ms | int8 speedup |
|---|---|---|---|
| 64×64 | 0.02 | 0.03 | 0.59× |
| 128×128 | 0.14 | 0.20 | 0.72× |
| 256×256 | 1.53 | 1.55 | 0.99× |
| 512×512 | 15.00 | 12.85 | 1.17× |

An honest result I want to highlight rather than hide: **scalar-widening INT8
does not automatically beat FP32 on a desktop-class core.** The compiler
vectorizes the fp32 inner loop with AVX2 FMAs, while the int8 path pays
sign-extension to int32 before multiplying. INT8 only pulls ahead when its 4×
smaller footprint starts winning cache residency (≥256²) — and on
cache-starved router-class SoCs that crossover comes much earlier. Closing the
gap on wide cores needs explicit int8 SIMD (`vpmaddubsw`/`vpdpbusd`, ARM
`sdot`) — top of the [BACKLOG](BACKLOG.md).

### End-to-end LeNet-shaped CNN (1×28×28, single-threaded)

| path | median latency | throughput | |
|---|---|---|---|
| FP32 reference | 0.473 ms | 2,113 inf/s | direct convolution |
| INT8 quantized | **0.162 ms** | **6,156 inf/s** | **2.9×** — im2col+GEMM, 29,008 B scratch |

Here INT8 wins decisively: conv layers at these shapes are memory-bound, and
the quantized path moves a quarter of the bytes *and* uses the GEMM lowering.

### Accuracy: INT8 vs FP32 golden (worst case over 8 random inputs)

| metric | value |
|---|---|
| max abs error | 0.3629 |
| cosine similarity | 0.99903 |

Test-enforced bounds: per-op ≤ 3 output quant steps and cosine > 0.999;
end-to-end ≤ 8 steps and cosine > 0.99 through five quantized layers.

Full results (with MAD and rep counts): run `./build/bench/edge_bench`, which
also writes `bench_results.json`. A rendered version lives in
[docs/](https://ericsson-kuma.github.io/edge-infer/).

## Testing

32 tests, all asserting against references rather than snapshots:

- quantize/dequantize round-trip error ≤ half a quant step; saturating clamp
  behaviour beyond the calibrated range; requantize rounding and saturation
  boundaries.
- INT8 GEMM variants **bit-exact** vs the naive reference (int32 accumulation
  is associative — so this is an equality, not a tolerance); FP32 variants
  within a K-scaled tolerance; worst-case accumulation overflow check.
- conv/fc/pool against hand-computed cases; strided+padded im2col taps.
- End-to-end: quantized graph tracks FP32 within the bounds above; ping-pong
  buffer reuse verified repeatable; scratch plan < 64 KiB.

CI runs the suite on gcc and clang (ubuntu-latest, `-Werror`) plus a bench
smoke run. Clang already caught one real C++17 portability bug (structured
bindings are not capturable by lambdas until C++20) that gcc silently accepts.

## Roadmap

Tracked in [BACKLOG.md](BACKLOG.md) — explicit int8 SIMD kernels, ARM NEON
port, per-channel quantization, operator fusion, a serialized model format,
and friends. Each item is scoped small enough to land as a single PR.

## License

[MIT](LICENSE)
