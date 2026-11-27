# moonbit-lowprec

Low-precision numeric types for MoonBit. Version 0.15.0 ships **IEEE 754
binary16**, **bfloat16**, **fixed-point**, **affine quantization**, the two
**8-bit floats**, a **GGUF container reader** down to the K-quant
super-blocks and a **writer** whose output llama.cpp's own Python tooling
reads back, with the **Q8_0 and Q4_0 quantizers** that fill it. It includes
correctly rounded **addition, subtraction,
multiplication and division** on both 16-bit float types — computed on the
significand with no floating-point instruction — **direct conversions** between
the formats, signal-to-noise reporting, published reference vectors and a
command-line inspector, everything checked against independent references.

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

The 8-bit formats are missing in the same way: neither `E4M3` nor `E5M2`
appears in those descriptions, and no listed module stores an 8-bit float as
a type a caller can hold. They matter because fp8 is what ONNX and the newer
inference stacks compress weights and activations to, and because E4M3 has a
corner that hand-written decoders get wrong: the largest finite value 448
shares an all-ones exponent with the NaN pattern, so the usual
"exponent all ones means special" rule is false for it.

The gap is not cosmetic. binary16 has a 5-bit exponent (range 6.1e-5 to
65504) and an 11-bit significand; bfloat16 keeps binary32's 8-bit exponent
but only 8 significant bits; fixed-point has no exponent at all and no NaN,
only a fixed scale and a hard edge at each end. None of these is
interchangeable with binary32 — and a type that can be stored but not added
is not yet a numeric layer, which is why the arithmetic is the largest part
of this release.

## What is implemented

### 16-bit arithmetic

`F16::add`, `F16::sub`, `F16::mul`, `F16::div` and the same four on `BF16`.

The two formats differ only in field widths, so one integer core serves both.
Each operand is decoded into a sign, an integer significand and a binary
exponent — `value = m * 2^e` — the significands are combined exactly in 64 bits, and the result is rounded **once** when it is encoded back into
the format. Consequences worth stating:

- the rounding sees every discarded bit, so there is no intermediate rounding
  to be fooled by, and the answer is the correctly rounded one
- ties go to even, the subnormal range is entered gradually, overflow goes
  to infinity rather than wrapping, and the sign of a zero result is
  preserved for `(-0) + (-0)`
- NaN and infinity propagate as IEEE 754 requires, including `infinity - infinity` and `infinity * 0` producing NaN
- no binary64 instruction is used anywhere, so the same bits come out on
  every backend and the code would still work on a target without a float
  unit

bfloat16 spans exponents that 64 bits cannot align: its smallest subnormal is
2^-133 while its largest value is near 2^128. When the exponent gap exceeds
fifty bits the smaller operand is below half an ulp of the larger, so the sum
rounds back to the larger operand; binary16 never reaches that gap (its whole
range fits in about forty bits), so the branch exists for one format only and
is covered by the sweep.

### `F16` — IEEE 754 binary16

- `F16::from_bits` / `bits`, field access, classification predicates
- `to_float`, `from_float` — round-to-nearest ties-to-even, overflow to
  infinity, gradual underflow to subnormal then zero
- `to_double`, `from_double`
- `add`, `sub`, `mul`, `div`
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

### `FP8` — 8-bit float formats

- `E4M3` — the OCP / ONNX `float8_e4m3fn`: one sign bit, four exponent bits
  with a bias of 7, three significand bits. It has no infinity: the largest
  finite value is 448 at `0x7E`, the smallest positive value is 2^-9, and
  only `0x7F` and `0xFF` are NaN. `0x7E` being finite is why the NaN test
  cannot be "exponent all ones".
- `E5M2` — the OCP / ONNX `float8_e5m2`: five exponent bits with a bias of
  15, two significand bits, infinities at `0x7C` / `0xFC`, NaN at the other
  six patterns, largest finite 57344.
- `from_bits`, `bits`, `sign_bit`, `exponent_field`, `mantissa_field`, and
  `is_nan` / `is_infinite` / `is_zero` / `is_subnormal` / `is_normal` /
  `is_finite` / `is_negative`.
- `to_float` never rounds: every 8-bit value is exactly representable in
  binary32. `from_float` rounds to nearest, ties to even, with gradual
  underflow. E4M3 saturates at 448 by default (the ONNX `Cast` behaviour);
  `from_float_overflow_nan` turns anything past 480 into NaN instead. E5M2
  overflows to infinity, and both formats flush below half the smallest
  subnormal.

### `GGUF` — container reading and writing

`parse_gguf` reads a GGUF v3 header: metadata pairs are skipped except
`general.alignment`, and the tensor table keeps each name, dimensions, ggml
type and payload offset. `parse_gguf_header` reads only the fixed 24-byte
prefix — magic, version, tensor count, metadata count — so a caller can
identify and size a file without holding it: useful when the payload is
hundreds of megabytes. `dequantize_tensor` turns a tensor into binary64 values:
`F32` is widened directly, while `Q4_0`, `Q5_0`, `Q8_0`, `Q2_K`, `Q3_K`,
`Q4_K`, `Q6_K` and `Q5_K` go block by block: 32 weights per block with one
f16 scale and 16, 22 or 34 bytes of codes, or 256 weights per K-quant
super-block — 84 bytes for `Q2_K`, 110 for `Q3_K`, 144 for `Q4_K`, 176 for
`Q5_K`, 210 for `Q6_K`. That is every K-quant type llama.cpp defines.

The K-quant layouts are the ones that speak for the format's difficulty.
`Q2_K` spends sixteen bytes on a four-bit scale and a four-bit min per
sixteen weights, then stores two bits per weight. `Q3_K` widens that to three
bits: the third lives in a shared mask, and a set bit means the weight keeps
its offset while a clear one subtracts four. `Q4_K` applies
`d * sc * q - dmin * m` per 32 weights, `Q5_K` adds a fifth bit per weight
taken from 32 shared bytes whose eight bits serve the four groups, and `Q6_K`
stores 6-bit signed weights biased by 32 and multiplies them by int8
sub-scales. Two of the scale packings are the ones llama.cpp writes into
twelve bytes — `get_scale_min_k4` (`ggml-quants.c:889`) for `Q4_K` and
`Q5_K`, and a sixteen-field six-bit layout for `Q3_K` — and both are checked
against tables written out by hand, so a transposed byte shows up as a wrong
scale rather than as a plausible number.

The layout itself is not unique to this project: `soehi/gguf` 0.1.3 decodes
`Q4_K` and `Q6_K`, to Float, inside its container reader. It has no `Q2_K`,
`Q3_K` or `Q5_K`, and the difference argued here is the output and the
surrounding layer as much as the block map: binary64 values that feed the
same report pipeline as the fixed-point and float formats.

The block readers are checked against two implementations written independently
of this one: `tools/emit_gguf_test.py`, built from the layouts alone, and the
maintained `gguf` Python package from the organisation that publishes
llama.cpp, which dequantizes the same blocks with a maximum difference of zero
on all 256 values of each of the five K-quant blocks. A one-off probe compared
all 256 values of `Q2_K` and `Q3_K` against the package before they were
reduced to the literals the tests keep. The fixture is a 1554-byte container
holding all nine types.

The payload never leaves the caller's byte array. That is deliberate: the
targets this project builds for have no file access, so a caller that does
read a file passes the bytes in and gets values back.

Full chain on a container carried as text:

```
$ moon run cmd/main --target wasm -- gguf <hex>
gguf v3  alignment 32  tensors 2
  token_embd.weight  kind 2  n 64  bytes 36
    re-quantized int8: step 0.031496062992125984  snr 62.606858853953355 dB  max 0.003937007874015741
  output_norm.weight  kind 0  n 8  bytes 32
    re-quantized int8: step 0.01968503937007874  snr 47.07591579591082 dB  max 0.007874015748031482
```

Those figures are the point: a tensor that arrived as Q4_0 still holds
62.6 dB after another int8 quantization, because the coarser format already
discarded the fine detail.

The same reader has now been run on a released model, not only on a
hand-built sample. `stories260K.gguf` (1,185,376 bytes, sha256
`270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d`) has
version 3, alignment 32, 19 metadata entries, 48 tensors and a payload that
starts at offset 14176 — the values this reader produces, the values
llama.cpp's own reader produces, and the values an independent Python
reader written for the test produces. Its first 24 bytes are in the test
suite, so the identification path is checked against a file this repository
did not create.

`build_gguf` runs the other way. It takes an array of `GgufTensorOut` —name,
dimensions, ggml type, payload — and produces a GGUF v3 container: the 24-byte
prefix, one entry per tensor, then each payload at an offset aligned to the
container's own alignment. Metadata is not written, so `kv_count` is zero and
`data_start` lands right after the table. The fixture's nine payloads survive
a write-and-read cycle byte for byte, in a container this project formed
rather than one it received.

Two quantizers fill those payloads. `quantize_q8_0` takes 32 values, stores
`amax / 127` as a binary16 scale and one signed byte per weight rounded half
away from zero — the rule llama.cpp applies, not the ties-to-even rule used in
the requantization path — and `quantize_q4_0` stores `amax / 8` with codes in
-8..7 biased by eight, the low nibble carrying weights 0..15 and the high
nibble 16..31.

The output is checked where it is hardest to fake: by handing the bytes to
`gguf`, the Python reader maintained by the organisation that publishes
llama.cpp. `tools/check_written.py` does exactly that and prints

```
210 bytes, magic b'GGUF'
  block_q8_0 type 8 shape [32] maxerr 0.00390625
  block_q4_0 type 2 shape [32] maxerr 0.0625
written container: read back by the official package, 0 errors
```

Both errors are exactly half a quantization step, which is the best a
round-to-nearest quantizer can promise, and they come out of the other
implementation's arithmetic on bytes this one wrote.

### Conversions between the formats

The formats convert directly, with integer arithmetic on the patterns rather
than widening to binary32 and narrowing again: `F16::from_bf16`,
`BF16::from_f16`, `F16::from_e4m3`, `F16::from_e5m2`, `E5M2::from_e4m3` and
`E4M3::from_e5m2`.

Two are exact and worth stating. bfloat16 to binary16 never rounds: seven
significand bits fit in ten, so only the exponent range matters — values past
65504 become infinity and everything under half of 2^-24 becomes zero. e5m2
to e4m3 is exact for normals because the significand grows from three bits to
four, and saturates at 448 because the range shrinks.

The other direction of each pair rounds once: binary16 to bfloat16 drops
three significand bits with ties to even, e4m3 to e5m2 drops one. All six
are checked by exhaustion against the binary64 path: every bfloat16 and
binary16 pattern (65282 and 63490 finite values), and all 256 patterns in
each direction for the fp8 pair.

## Correctness evidence

`moon test` runs eighty-eight tests on `wasm`, `wasm-gc` and `js`, all passing.

Exhaustive checks, not samples:

1. **binary16 round-trip** over all 65536 patterns — 63490 non-NaN checked.
2. **bfloat16 round-trip** over all 65536 patterns — 65282 non-NaN checked.
3. **binary16 arithmetic sweep**: all 65536 patterns against twelve probes,
   add, multiply, divide and subtract — 3,145,728 results compared against a
   higher-precision path.
4. **bfloat16 arithmetic sweep**: the same 65536 patterns against the same
   probes and the same four operations — another 3,145,728 results.
5. **Reference table**: 171 operand pairs × four operations × two formats,
   generated from exact rational arithmetic and one rounding. The binary16
   half of it also agrees with numpy's float16 on all 684 results.
6. **Fixed-point multiplication**: all 65536 operand pairs from the signed
   8-bit range × five rounding modes = 327680 results against an
   independent `Double` reference.
7. **Widening multiply**: the same 65536 pairs must produce the raw product
   exactly.
8. **Saturation and wrap** over every one of the 65536 raw values in a
   16-bit field.
9. **Requantization**: all 256 int8 codes against a `Double` computation,
   with the exact-match count asserted.
10. **Rounding bound**: 990 signal points across -0.99..0.99 must stay
    within half a step of the nearest representable value, and truncation
    must show a strictly larger maximum error than nearest-even on the same
    points.
11. **FP8 table**: all 256 patterns of each format compared value by value
    against a table decoded with exact rational arithmetic by an independent
    implementation, then every finite pattern narrowed back to its own bits;
    the positive half of each format must also be strictly increasing with
    the pattern, and the NaN counts (2 for E4M3, 6 for E5M2) are asserted.
12. **Container write-back**: the nine payloads of the fixture are copied into
    a fresh container built here and read back — names, types, sizes and bytes
    unchanged. The two quantizers are measured against half a step on 32
    values, and the bytes they produce are read by the official Python
    package with maximum errors of exactly 0.00390625 and 0.0625.

Targeted tests cover hand-checked rounding directions, division, per-channel
parameters, integer dot products, clamp edges, SNR agreement with an
independent `Double` computation, all sixteen Q4_0 codes, IEEE special values
(NaN, infinity, overflow, signed zero), the header of a released model, and
truncated containers that must be refused. For the 8-bit formats they pin the
448 boundary at 480, the infinity that E4M3 cannot store, and the tie at
1.5 × 2^-9 where ties-to-even picks the even pattern.

Two independent routes produce the arithmetic answers: the integer core and
the `Double`-mediated path. They agree everywhere in the sweeps above, and
during development the disagreement they exposed was a real defect —
`(-0) + (-0)` losing its sign. The bfloat16 wide-gap branch was likewise found
by the sweep, not by inspection: an aligned shift of 126 bits silently
produced zero.

## Platform notes

`Int` in this dialect is a signed 32-bit word. That constrains two things
and they are stated rather than hidden:

- fixed-point multiplication supports fields up to 16 bits, where the
  product stays below 2^30; wider accumulation goes through `mul_raw` or
  `dot_i8`
- the requantization product reaches about 2^38, so `requantize` forms it
  in `Int64` before rounding

The arithmetic core works in 64 bits and needs no wider type; the bfloat16
exponent gap is handled by the half-ulp rule described above rather than by
extended precision.

`moon check` reports no errors and no warnings on compiler 0.1.20261023. On
0.1.20261104 the same sources build cleanly but the newer compiler marks
`Double::from_int` and `Int64::from_int` as deprecated, which is a naming
notice rather than a correctness issue.

## Usage

```moonbit
let a = F16::from_float(1.5)
assert_eq(a.to_float(), 1.5)
assert_eq(a.add(a).bits(), 0x4100)        // 3.0

let b = BF16::from_float(1.0)
assert_eq(b.div(b).bits(), 0x3F80)        // 1.0

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

$ moon run cmd/main --target wasm -- arith f16 0x3C00 0x4200
f16 0x3c00 and 0x4200  (1, 3)
  sum        0x4400  4
  difference 0xc000  -2
  product    0x4200  3
  quotient   0x3555  0.33325195

$ moon run cmd/main --target wasm -- fp8 e4m3 0x7e
bits=126 sign=0 exp=15 mant=6 class=normal value=448
```

The `arith` block is the shortest check of the 16-bit arithmetic: 1 and 3 give
4, -2, 3 and the correctly rounded 1/3, and every value can be confirmed by
hand. `fp8 e4m3 0x7e` shows the odd corner of E4M3: an exponent of all ones
with a significand other than seven is a finite 448, not a NaN.

`requant` prints the integer multiplier and shift it derived, so the
quantization arithmetic is inspectable rather than opaque. `q4_0` takes a
sixteen-bit scale and sixteen packed bytes, `gguf` takes a whole container
as hex, `emit-gguf` writes a two-tensor container and prints its hex so a
reader elsewhere can open it, `verify` re-checks the published vectors,
`selftest` runs seven built-in checks.

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

`fp_vectors.mbt` goes further: 171 operand pairs per format with the expected
result of each of the four operations, so the table checks arithmetic and not
only widening. `verify_vectors()` returns `(checked, failed)`,
`moon run cmd/main -- verify` prints `vectors 60/60 reproduce`, and a test
fails if any row stops reproducing.

`fp8_vectors.mbt` publishes the other direction: all 256 patterns of `E4M3`
and all 256 of `E5M2`, each with its exact value and a marker for NaN or
infinity. A caller porting either format to another language can compare
against those rows without running this code.

## Roadmap

- 0.11: `E4M3` and `E5M2`, each checked pattern by pattern against a rational
  table and round-tripped
- 0.12: `Q4_K`, `Q6_K` and `Q5_K` super-blocks, the formats behind the
  Q4_K_M, Q5_K_M and Q6_K files llama.cpp ships, cross-checked against the
  `gguf` package
- 0.13: direct conversions between the formats, exact where the target holds
  more significand bits, and checked by exhaustion
- 0.14: `Q2_K` and `Q3_K`, so every K-quant type llama.cpp defines reads here
- 0.15: `build_gguf`, `quantize_q8_0` and `quantize_q4_0`, so the bytes a
  reader consumes can also be produced and handed back to llama.cpp's tooling
- later: fp8 dot products, and a writer for the K-quant blocks when a caller
  needs to emit them rather than because the format list has gaps

## License

Apache-2.0.
