# Changelog

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
