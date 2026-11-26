# Changelog

## 0.7.0 — 2026-11-26

Published reference vectors.

- `vectors.mbt` exports `f16_vectors`, `bf16_vectors`, `quant_vectors` and
  `fixed_vectors`: pattern plus the exact binary64 value it must produce,
  sixty rows in total.
- `verify_vectors()` returns `(checked, failed)`; `cmd/main verify` prints
  `vectors 60/60 reproduce`.
- Two tests: the table must reproduce in full, and it must still contain the
  extremes of each type.
- An adjacent module was re-checked at source level:
  `matrixlt/moon-onnx` does implement per-tensor int8/uint8 quantization, so
  the gap is narrower than a description scan suggests. The gap claims in
  `docs/coverage.md` and the README now say what is actually absent
  (binary16/bfloat16 types, fixed-point, per-channel scale, integer
  requantization, error reporting) instead of "no int8 anywhere".

## 0.6.0 — 2026-11-26

Quantization error reporting.

- `quantization_report(values, params, mode)` returns `QuantReport` with
  `max_abs_error`, `mean_abs_error`, `snr_db` and `count`.
- `snr_db` is `10 * log10(signal energy / error energy)`; exact reproduction
  reports positive infinity instead of a sentinel.
- `cmd/main report` runs a 256-sample sine through all four rounding modes
  so the cost of truncation is visible as a number (about 5.5 dB here).
- Six new tests, thirty-seven total, still passing on the three backends.

## 0.5.0 — 2026-11-26

A command-line inspector for the types above.

- `cmd/main` executable with `f16`, `bf16`, `fixed`, `quant`, `requant`,
  `q4_0` and `selftest` subcommands, built on `@env.args()` only.
- `requant` prints the derived multiplier and shift next to the resulting
  code, so the integer path can be inspected instead of assumed.
- `Rounding` and `Overflow` constructors are now `pub(all)`: callers
  outside the package could previously only reach them through
  `Rounding::from_int` / `Overflow::from_int`, which made the API awkward to
  use from another package.
- No change to the library logic; all thirty-one tests still pass on the
  three backends.

## 0.4.0 — 2026-11-26

Affine quantization on top of the fixed-point core.

- `QuantParams` with `int8`/`uint8` ranges, `quantize`, `dequantize`.
- `RequantScale::from_scales` normalizes `scale_in / scale_out` into a
  31-bit multiplier plus a shift.
- `requantize` runs entirely on integers: 64-bit product, one rounding, one
  clamp.
- Per-channel parameters, 64-bit integer dot product.
- `ggml_q4_0_dequantize` reads a real GGML Q4_0 block, wiring `F16` into a
  container format.
- `Fixed` multiplication restricted to fields up to 16 bits, and the
  requantization product widened to 64 bits, because `Int` is a 32-bit word
  in this dialect. Both limits are documented rather than hidden.
- Thirty-one tests passing on wasm, wasm-gc and js.
## 0.3.0 — 2026-11-26

Fixed-point arithmetic.

- `Fixed` type: `raw * 2^-frac` in a `width`-bit field, stored in `Int`.
- Five rounding modes (`TowardZero`, `TowardNegative`, `TowardPositive`,
  `NearestEven`, `NearestAway`) defined on the magnitude with the sign
  restored, so negatives follow the named direction.
- Two overflow modes (`Saturate`, `Wrap`), applied only when a result is
  settled into the field.
- `add`, `sub`, `neg`, `abs`, `mul`, `div`, `rescale`, plus `mul_raw` for
  exact multi-term accumulation.
- Exhaustive tests: 327680 multiplication results against an independent
  `Double` reference, exact widening multiply over all 8-bit pairs, and
  saturation/wrap over every 16-bit raw value.
- `Rounding::from_int` and `Overflow::from_int` for command-line parsing.
- Twenty-three tests passing on wasm, wasm-gc and js.

## 0.2.0 — 2026-11-26

bfloat16 joins binary16.

- `BF16` type, raw pattern storage, same surface as `F16`.
- Widening to binary32 is a shift and exact for every pattern.
- Narrowing with round-to-nearest ties-to-even, plus `from_float_truncate`
  for the drop-the-tail hardware conversion. NaN payloads that would
  truncate into infinity are re-quieted.
- Exhaustive round-trip over all 65536 patterns (65282 non-NaN) and
  monotonicity over the positive finite range.
- `F16` constants moved to `F16::name` form so both types present the same
  API shape.
- Fourteen tests passing on wasm, wasm-gc and js.

## 0.1.0 — 2026-11-26

First cut: IEEE 754 binary16.

- `F16` type stored as a raw 16-bit pattern.
- Exact widening to binary32 for every pattern including subnormals;
  narrowing with round-to-nearest, ties-to-even, overflow to infinity and
  gradual underflow.
- Eight tests, two of them exhaustive over all 65536 patterns.
