# Changelog

## 0.1.0 — 2026-11-26

First cut: IEEE 754 binary16.

- `F16` type stored as a raw 16-bit pattern.
- Exact widening to binary32 (`to_float`) for every pattern including
  subnormals; narrowing with round-to-nearest, ties-to-even, overflow to
  infinity and gradual underflow (`from_float`).
- Classification predicates.
- Eight tests, two of them exhaustive over all 65536 patterns; passing on
  wasm, wasm-gc and js.
