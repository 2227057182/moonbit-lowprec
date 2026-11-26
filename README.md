# moonbit-lowprec

Low-precision numeric types for MoonBit. Version 0.3.0 ships **IEEE 754
binary16**, **bfloat16** and **fixed-point** arithmetic with explicit
rounding and overflow semantics; quantized operators follow in 0.4.

## Why this exists

MoonBit already has floating point, but it has no *low-precision* numeric
layer. A scan of the 3975 modules published on mooncakes.io (2026-11-26)
returns **zero** matches for `fixed point`, `int8`, `int16`, `f16`,
`fp16` and `bfloat16`. The nearest neighbours cover different ground:

| Module | What it covers | What it does not |
|---|---|---|
| `Luna-Flow/floating` | arbitrary-precision binary/decimal and interval arithmetic | 16-bit formats, fixed-point, quantization |
| `moonbit-community/softfloat` | IEEE-754 software float emulation | binary16/bfloat16 as types |
| `soehi/gguf` | reads GGUF containers, dequantizes ggml blocks | the numeric types those blocks represent |
| `matrixlt/onnx` | ONNX graph execution | an exposed low-precision type layer |

The gap is not cosmetic. binary16 has a 5-bit exponent (range 6.1e-5 to
65504) and an 11-bit significand; bfloat16 keeps binary32's 8-bit exponent
but only 8 significant bits; fixed-point has no exponent at all and no NaN,
only a fixed scale and a hard edge at each end. None of these is
interchangeable with binary32.

## What is implemented

### `F16` — IEEE 754 binary16

- `F16::from_bits` / `bits`, field access, classification predicates
- `to_float`, `from_float` — round-to-nearest ties-to-even, overflow to
  infinity, gradual underflow to subnormal then zero
- `to_double`, `from_double`
- constants: `min_subnormal`, `min_normal`, `max_value`, `infinity`, `nan`,
  `epsilon`, `precision`

### `BF16` — bfloat16

- same surface, plus `from_float_truncate` for the hardware conversion that
  drops the low 16 bits
- widening is a shift and loses nothing; finite inputs cannot overflow by
  magnitude because the exponent range matches binary32

### `Fixed` — signed fixed-point

A value is `raw * 2^-frac` in a `width`-bit field, stored in an `Int`. The
storage is wider than any supported field, so the exact product of two
values always fits and no intermediate rounding is forced.

- five rounding modes: `TowardZero`, `TowardNegative`, `TowardPositive`,
  `NearestEven`, `NearestAway`
- two overflow modes: `Saturate`, `Wrap`
- `add`, `sub`, `neg`, `abs`, `mul`, `div`, `rescale`
- `mul_raw` returns the exact product for multi-term accumulation
- `from_int`, `from_raw`, `to_double`, `is_valid`

Rounding is defined on the magnitude with the sign restored, so negative
inputs follow the named direction instead of inheriting C's
truncation-toward-zero by accident.

## Correctness evidence

`moon test` runs twenty-three tests on `wasm`, `wasm-gc` and `js`, all
passing in about half a second per backend.

Exhaustive checks, not samples:

1. **binary16 round-trip** over all 65536 patterns — 63490 non-NaN checked.
2. **bfloat16 round-trip** over all 65536 patterns — 65282 non-NaN checked.
3. **binary16 and bfloat16 monotonicity** over the positive finite range.
4. **Classification** for both float types against the raw fields on all
   65536 patterns.
5. **Fixed-point multiplication**: all 65536 operand pairs from the signed
   8-bit range × five rounding modes = 327680 results, each compared with
   an independent `Double` reference.
6. **Widening multiply**: the same 65536 pairs at 8 fractional bits
   multiplied to 16 fractional bits must produce the raw product exactly —
   the double-width multiply-accumulate case.
7. **Saturation and wrap**: every one of the 65536 raw values in a 16-bit
   field, adding one, under both overflow modes.

Targeted tests cover hand-checked rounding directions (positive, negative,
exact halves), division, requantization, and negation at the field edge.

## Usage

```moonbit
let a = F16::from_float(1.5)
let b = BF16::from_bits(0x3F80) // 1.0
assert_eq(a.to_float(), 1.5)
assert_eq(b.to_float(), 1.0)

// 1.5 * 1.5 = 2.25, settled on the integer grid, nearest even.
let x = Fixed::from_raw(3, 1, 16)
assert_eq(x.mul(x, 0, NearestEven, Saturate).raw(), 2)
```

## Roadmap

- 0.4: quantization operators — affine scale/zero-point, per-tensor and
  per-channel, quantize/dequantize/requantize, integer matrix multiply,
  checked against a GGUF block

Each stage keeps the same rule: reference vectors computed independently,
checked bit for bit in the repository, no floating tolerance in tests.

## License

Apache-2.0.
