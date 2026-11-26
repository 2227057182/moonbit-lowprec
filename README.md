# moonbit-lowprec

Low-precision numeric types for MoonBit. Version 0.7.0 ships **IEEE 754
binary16**, **bfloat16**, **fixed-point** and **affine quantization**, with
signal-to-noise reporting, published reference vectors and a command-line
inspector, everything checked against independent references.

## Why this exists

MoonBit already has floating point, but it has no *low-precision* numeric
layer. A scan of the 3975 module descriptions published on mooncakes.io
(2026-11-26) returns **zero** matches for `fixed point`, `int16`, `f16`,
`fp16` and `bfloat16`. The nearest neighbours cover different ground:

| Module | What it covers | What it does not |
|---|---|---|
| `Luna-Flow/floating` | arbitrary-precision binary/decimal and interval arithmetic | 16-bit formats, fixed-point, quantization |
| `moonbit-community/softfloat` | IEEE-754 software float emulation | binary16/bfloat16 as types |
| `soehi/gguf` | reads GGUF containers, dequantizes ggml blocks | the numeric types those blocks represent |
| `matrixlt/moon-onnx` | ONNX graph execution; per-tensor int8/uint8 quantization operators | binary16/bfloat16, fixed-point, per-channel parameters, requantization, any exposed type |

`matrixlt/moon-onnx` is the one place int8 already appears: its
`ml_kernels/quant` package implements `QuantizeLinear`-style per-tensor
quantization with round-half-to-even. That is an operator inside a graph
runner, not a numeric type, and it has no float16, bfloat16, per-channel
scale, requantization path or error reporting. Those are the parts this
project supplies.

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

A value is `raw * 2^-frac` in a `width`-bit field, stored in an `Int`.

- five rounding modes: `TowardZero`, `TowardNegative`, `TowardPositive`,
  `NearestEven`, `NearestAway`
- two overflow modes: `Saturate`, `Wrap`
- `add`, `sub`, `neg`, `abs`, `mul`, `div`, `rescale`, `mul_raw`
- rounding is defined on the magnitude with the sign restored, so negative
  inputs follow the named direction instead of inheriting truncation toward
  zero

### Quantization

- `QuantParams` — affine `x = (q - zero_point) * scale`, with `int8` and
  `uint8` constructors
- `quantize`, `dequantize`, `quantize_channel`, `dequantize_channel`
- `RequantScale::from_scales` derives `scale_in / scale_out` as an integer
  multiplier normalized into `[2^30, 2^31)` plus a right shift
- `requantize` moves between domains with no float in the middle: the
  product is formed in 64 bits, rounded once, clamped once
- `dot_i8` accumulates in 64 bits
- `ggml_q4_0_dequantize` reads a real GGML Q4_0 block: 32 weights from 16
  packed bytes and a binary16 scale — the point where `F16` meets a
  container format
- `quantization_report` measures a configuration on a given signal:
  `max_abs_error`, `mean_abs_error` and `snr_db` =
  `10 * log10(signal energy / error energy)`, reported as positive infinity
  when nothing is lost

## Correctness evidence

`moon test` runs thirty-nine tests on `wasm`, `wasm-gc` and `js`, all
passing.

Exhaustive checks, not samples:

1. **binary16 round-trip** over all 65536 patterns — 63490 non-NaN checked.
2. **bfloat16 round-trip** over all 65536 patterns — 65282 non-NaN checked.
3. **binary16 and bfloat16 monotonicity** over the positive finite range.
4. **Classification** for both float types on all 65536 patterns.
5. **Fixed-point multiplication**: all 65536 operand pairs from the signed
   8-bit range × five rounding modes = 327680 results against an
   independent `Double` reference.
6. **Widening multiply**: the same 65536 pairs must produce the raw product
   exactly.
7. **Saturation and wrap** over every one of the 65536 raw values in a
   16-bit field.
8. **Requantization**: all 256 int8 codes against a `Double` computation,
   with the exact-match count asserted.
9. **Rounding bound**: 990 signal points across -0.99..0.99 must stay
   within half a step of the nearest representable value, and truncation
   must show a strictly larger maximum error than nearest-even on the same
   points.

Targeted tests cover hand-checked rounding directions, division,
per-channel parameters, integer dot products, clamp edges, SNR agreement
with an independent `Double` computation, and all sixteen Q4_0 codes.

## Platform notes

`Int` in this dialect is a signed 32-bit word. That constrains two things
and they are stated rather than hidden:

- fixed-point multiplication supports fields up to 16 bits, where the
  product stays below 2^30; wider accumulation goes through `mul_raw` or
  `dot_i8`
- the requantization product reaches about 2^38, so `requantize` forms it
  in `Int64` before rounding

## Usage

```moonbit
let a = F16::from_float(1.5)
assert_eq(a.to_float(), 1.5)

let x = Fixed::from_raw(3, 1, 16) // 1.5
assert_eq(x.mul(x, 0, NearestEven, Saturate).raw(), 2)

let p = QuantParams::int8(0.05, 0)
assert_eq(dequantize(quantize(1.25, p, NearestEven), p), 1.25)
```

## Inspector

`cmd/main` is a small executable that takes bit patterns and printed values
on the command line:

```
$ moon run cmd/main --target wasm -- f16 0x0002
bits=2 sign=0 exp=0 mant=2 class=subnormal value=1.1920929e-7

$ moon run cmd/main --target wasm -- bf16 0x3F80
bits=16256 sign=0 exp=127 mant=0 class=normal value=1

$ moon run cmd/main --target wasm -- fixed 3 1 16
raw=3 frac=1 width=16 valid=true value=1.5

$ moon run cmd/main --target wasm -- requant 25 0.05 0 0.01 128
multiplier=1342177280 shift=28 code=253 value=1.25
```

`requant` prints the integer multiplier and shift it derived, so the
quantization arithmetic is inspectable rather than opaque. `q4_0` takes a
sixteen-bit scale and sixteen packed bytes; `verify` re-checks the published
vectors; `selftest` runs seven built-in checks.

`report` runs the same sine signal through the four rounding modes, which
is the shortest way to see what rounding costs:

```
$ moon run cmd/main --target wasm -- report
signal: 256 sine samples, amplitude 0.98 (inside int8 at this scale)
int8, scale 0.0078125; the clamp is therefore not the limiting error
nearest-even     snr=49.64783777764896 dB  max=0.00387994390673585  mean=0.0019332765206122346
toward-zero      snr=44.28974511105994 dB  max=0.007689669576354974  mean=0.003555764862619121
toward-negative  snr=43.721530746038056 dB  max=0.007782736282212155  mean=0.003875732421875004
toward-positive  snr=43.67102652611996 dB  max=0.00781249999999988  mean=0.003906249999999996
```

Nearest-even holds the maximum error at half a step and gains about five
and a half decibels over the truncating modes, which is the expected
result: truncation loses up to a full step instead of half.

## Published vectors

`vectors.mbt` exports the check table itself, so another implementation can
compare against this one without importing it:

```moonbit
f16_vectors  : Array[(Int, Double)] // 24 binary16 patterns
bf16_vectors : Array[(Int, Double)] // 14 bfloat16 patterns
quant_vectors: Array[(Int, Double)] // 11 int8 codes at scale 0.0078125
fixed_vectors: Array[(Int, Double)] // 11 raw values at frac 3
```

Each row is the pattern and the exact binary64 value it must widen to. The
table pins both ends of every type — the smallest binary16 subnormal, the
largest binary16 finite value, the largest bfloat16 finite value, both int8
codes, and the edges of a 16-bit fixed field. `verify_vectors()` returns
`(checked, failed)`, `moon run cmd/main -- verify` prints
`vectors 60/60 reproduce`, and a test fails if any row stops reproducing.

## Roadmap

- 0.8: numeric comparison against `matrixlt/moon-onnx`'s per-tensor int8
  operators, so the two implementations can be cross-checked on the one
  overlap they share

## License

Apache-2.0.
