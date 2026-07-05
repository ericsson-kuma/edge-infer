# Backlog

Small, self-contained improvements, roughly in priority order. Each is scoped
to land as one PR with tests and (where relevant) before/after bench numbers.

1. **Explicit int8 SIMD inner kernel (x86).** Use `vpmaddubsw`+`vpmaddwd` (or
   `vpdpbusd` where AVX-VNNI exists) in the blocked GEMM inner loop, behind a
   runtime CPUID check with the current scalar path as fallback. Goal: flip
   the INT8-vs-FP32 table at ≤256² sizes. Acceptance: bit-exact vs naive i8.

2. **ARM NEON path.** Port the blocked int8 kernel to NEON (`sdot` on
   ARMv8.2+, `smlal` fallback) — the actual target ISA for router/AP SoCs.
   Needs a QEMU-based cross-compile CI job (aarch64-linux-gnu + qemu-user).

3. **Per-channel weight quantization for conv/fc.** One scale per output
   channel: requantize takes a multiplier vector instead of a scalar.
   Measure accuracy delta on the e2e model vs per-tensor.

4. **Integer-only requantize.** Replace the float multiplier with fixed-point
   (int32 multiplier + right shift, round-to-nearest-even), matching what
   integer-only NPUs/DSPs need. Assert max 1-step divergence from the float
   reference across the int32 range.

5. **Multi-threaded GEMM.** Partition M across a small thread pool
   (std::thread, no dependency), with a size cutoff so small matrices stay
   single-threaded. Bench scaling on 2/4/8 threads.

6. **Operator fusion: conv+ReLU and fc+ReLU.** Clamp during requantize
   instead of a separate pass over the activation buffer; removes one full
   read+write of every activation. Measure e2e latency delta.

7. **Serialized model format.** A tiny flat binary format (header + qparams +
   int8 weight blobs) with a loader, so a model calibrated on a host can be
   shipped to a target without the FP32 weights. Include an endianness check.

8. **Packed-B GEMM.** Pre-pack B into the blocked layout once at quantize()
   time (weights never change), eliminating the strided panel reads that
   remain in the blocked kernel. This is where the 512² int8 number should
   pull further ahead of fp32.

9. **avgpool2d + softmax (int8-friendly).** The two ops missing for a
   MobileNet-style head; softmax via LUT over the int8 domain.

10. **Calibration over a batch.** Replace single-input calibration with
    max-over-N (and optionally percentile clipping) to reduce sensitivity to
    an unlucky calibration sample; expose it as `quantize(span<Tensor>)`.

11. **Static-plan introspection.** Dump the memory plan (per-layer activation
    sizes, live ranges, scratch high-water mark) as JSON for the docs page —
    useful when deciding whether a model fits an MCU's SRAM.

12. **Benchmark CO2 guardrail: pinned-frequency mode.** Document (and detect)
    CPU frequency scaling; warn when `scaling_governor != performance` so
    published numbers aren't skewed by a throttled host.
