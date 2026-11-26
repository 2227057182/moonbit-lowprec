# moonbit-lowprec

Low-precision numeric types for MoonBit. Version 0.2.0 ships **IEEE 754
binary16** and **bfloat16** as first-class types with exact bit-level
conversions; fixed-point and quantized arithmetic follow in the same line.

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
but only 8 significant bits. Neither is interchangeable with binary32, and
code that stores model weights or sensor samples needs the 16-bit
representation to be exact rather than approximated.

## What is implemented

Both types store the raw bit pattern, so every operation is reproducible on
every backend and no host float is involved until you ask for one.

**`F16` — IEEE 754 binary16**

- `F16::from_bits` / `bits`
- field access: `sign_bit`, `exponent_field`, `mantissa_field`
- classification: `is_nan`, `is_infinite`, `is_zero`, `is_subnormal`,
  `is_normal`, `is_negative`, `is_finite`
- `to_float`, `from_float` — round-to-nearest, ties-to-even; overflow to
  infinity, gradual underflow to subnormal then zero
- `to_double`, `from_double`
- constants: `min_subnormal`, `min_normal`, `max_value`, `infinity`, `nan`,
  `epsilon`, `precision`

**`BF16` — bfloat16**

- the same surface, plus `from_float_truncate` for the hardware conversion
  that simply drops the low 16 bits
- widening is a shift and loses nothing; bfloat16 shares binary32's exponent
  range, so finite inputs never overflow by magnitude

## Correctness evidence

`moon test` runs fourteen tests on `wasm`, `wasm-gc` and `js`, all passing.

Four are exhaustive rather than spot checks:

1. **binary16 round-trip over all 65536 patterns** — widen to binary32 and
   narrow back; 63490 non-NaN patterns checked, 0 mismatches.
2. **bfloat16 round-trip over all 65536 patterns** — 65282 non-NaN patterns
   checked, 0 mismatches.
3. **binary16 monotonicity** over patterns 0x0000..0x7BFF — catches a
   mis-normalised subnormal that spot values miss.
4. **bfloat16 monotonicity** over patterns 0x0000..0x7F7F.

Classification predicates are checked against the raw fields on all 65536
patterns for both types. Rounding, overflow, underflow, and both
ties-to-even directions have targeted tests, as does the NaN payload that
would otherwise become infinity under truncation.

## Usage

```moonbit
let a = F16::from_float(1.5)
let b = BF16::from_bits(0x3F80) // 1.0
assert_eq(a.to_float(), 1.5)
assert_eq(b.to_float(), 1.0)
```

## Roadmap

- 0.3: fixed-point `Qm.n` — saturating and wrapping modes, five rounding
  modes, double-width multiply-accumulate
- 0.4: quantization operators — affine scale/zero-point, per-tensor and
  per-channel, quantize/dequantize/requantize, integer matrix multiply,
  checked against a GGUF block

Each stage keeps the same rule: reference vectors generated offline, checked
bit for bit in the repository, no floating tolerance in tests.

## License

Apache-2.0.
