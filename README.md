# moonbit-lowprec

Low-precision numeric types for MoonBit. Version 0.1.0 ships **IEEE 754
binary16 (half precision)** as a first-class type with exact bit-level
conversions; bfloat16, fixed-point and quantized arithmetic follow in the
same line.

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

The gap is not cosmetic: binary16 is a distinct type with a 5-bit exponent
(range 6.1e-5 .. 65504) and 11-bit significand. Values that fit binary16 are
not interchangeable with binary32, and code that stores model weights or
sensor samples needs the 16-bit representation to be exact. Reading quantized
data without owning the types forces every consumer to re-derive the same
bit manipulation.

## What is implemented

`F16` stores the raw 16-bit pattern, so every operation is reproducible on
every backend and no host float is involved until you ask for one.

- `F16::from_bits` / `bits`
- field access: `sign_bit`, `exponent_field`, `mantissa_field`
- classification: `is_nan`, `is_infinite`, `is_zero`, `is_subnormal`,
  `is_normal`, `is_negative`, `is_finite`
- `to_float`, `from_float` (round-to-nearest, ties-to-even; overflow to
  infinity, gradual underflow to subnormal, then zero)
- `to_double`, `from_double`
- constants: `min_subnormal`, `min_normal`, `max_value`, `infinity`, `nan`,
  `epsilon`, `precision`

## Correctness evidence

`moon test` runs eight tests on `wasm`, `wasm-gc` and `js`, all passing.

Two of them are exhaustive rather than spot checks:

1. **Round-trip over all 65536 patterns.** Every non-NaN binary16 pattern is
   widened to binary32 and narrowed back; the result must equal the original
   bits. 63490 patterns checked, 0 mismatches.
2. **Monotonicity over all positive finite values.** Walking bit patterns
   0x0000..0x7BFF, the widened value must never decrease — this catches a
   mis-normalised subnormal that a handful of spot values would miss.

Classification predicates are also checked against the raw fields on all
65536 patterns. Rounding, overflow and underflow have targeted tests
including both ties-to-even cases.

## Usage

```moonbit
let x = F16::from_float(1.5)
let y = F16::from_bits(0x3C00) // 1.0
assert_eq(x.to_float(), 1.5)
assert_eq(y.bits(), 0x3C00)
```

## Roadmap

- 0.2: bfloat16 (8-bit exponent, truncating conversion)
- 0.3: fixed-point `Qm.n` with saturating and wrapping modes, five rounding
  modes, double-width multiply-accumulate
- 0.4: quantization operators — affine scale/zero-point, per-tensor and
  per-channel, quantize/dequantize/requantize, integer matrix multiply

Each stage keeps the same rule: reference vectors generated offline, checked
bit for bit in the repository, no floating tolerance in tests.

## License

Apache-2.0.
