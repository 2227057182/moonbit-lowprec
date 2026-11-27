# Changelog

## 0.13.0 — 2026-11-27

Direct conversions between the formats, in integer arithmetic.

- `convert.mbt` adds `F16::from_bf16`, `BF16::from_f16`, `F16::from_e4m3`,
  `F16::from_e5m2`, `E5M2::from_e4m3` and `E4M3::from_e5m2`. None of them
  goes through binary32: the sign moves as a bit, the exponent is added to or
  rebased from a bias, and the significand is shifted once, with ties to even
  when bits are dropped.
- bfloat16 to binary16 is the one direction that never rounds, because seven
  significand bits fit in ten. Only the exponent range changes: values above
  65504 become infinity (28674 patterns), and everything below half of 2^-24
  becomes zero (26112 patterns, the 254 subnormals among them).
- e5m2 to e4m3 is exact for normals for the same reason in reverse, saturates
  at 448 (60 patterns: both infinities, the two 448s and 56 larger values) and
  sends the three subnormals to zero.
- The exhaustive check found a real defect in this release. `round_shift_even`
  took a shift of 64 or more straight to a right shift, but MoonBit's `Int64`
  shifts wrap modulo the word width, so `1 << 71` became `1 << 7` and a
  bfloat16 value near 1.6e-27 came out as the smallest binary16 subnormal
  instead of zero. Shifts of 64 or more now return zero with the reason written
  down, and the exhaustive bfloat16 to binary16 test covers the case.
- Seven new tests, 83 in all: every bfloat16 pattern (65536, NaN compared by
  class only) and every binary16 pattern against the binary64 path, the
  positive bfloat16 half checked for monotonicity with zero backward steps,
  all 256 patterns of e4m3 and e5m2 into binary16, the 256 e4m3 to e5m2
  patterns, and the 256 e5m2 to e4m3 patterns with the saturation count fixed
  at 60.
- The CLI gained `convert <from> <to> <bits>`, covering the six integer
  conversions plus binary16 into either fp8 format.

## 0.12.0 — 2026-11-27

The three K-quant super-blocks, the formats behind the Q4_K_M, Q5_K_M and
Q6_K files llama.cpp ships.

- `dequantize_tensor` handles `Q4_K` (kind 12), `Q5_K` (kind 13) and `Q6_K`
  (kind 14). A K-quant block covers 256 weights rather than 32, so
  `byte_size` grows a super-block branch: 144 bytes for `Q4_K`, 176 for
  `Q5_K`, 210 for `Q6_K`.
- `Q4_K`: f16 `d`, f16 `dmin`, twelve packed bytes, then 128 bytes of 4-bit
  weights. The twelve bytes hold eight six-bit sub-scales and eight six-bit
  mins; the second four of each take their high two bits from bytes that
  already carry the first four. Each 32 weights become
  `d * sc * q - dmin * m`, with the sub-block index advancing every 64 weights.
  The unpacking follows llama.cpp's `get_scale_min_k4` bit for bit.
- `Q5_K`: the same twelve packed bytes and the same per-32 formula, with a
  fifth bit per weight stored in 32 shared bytes. Group `g` reads bit `2g` of
  the shared byte for the low nibbles and bit `2g+1` for the high nibbles,
  which is why the llama.cpp kernel shifts a mask rather than advancing the
  pointer.
- `Q6_K`: 128 low nibbles, 64 bytes of high bits, sixteen int8 sub-scales and
  an f16 scale. Weights are 6-bit signed-biased-by-32, sub-scales are signed,
  and the four weight lanes of each 32-value group take scales at `is + 0/2/4/6`.
- The fixture container grew to 1264 bytes with one block of each of the seven
  supported types, so the whole chain is exercised through the same reader a
  caller uses.
- Two independent implementations agree with this one. `tools/emit_gguf_test.py`
  is written from the block layouts alone, and the maintained `gguf` Python
  package (from the organisation that publishes llama.cpp) dequantizes the
  same blocks to the same 256 values with a maximum difference of zero.
- `gguf_kquant_wbtest.mbt`: the twelve-byte scale unpacking against a hand
  written table, both bit paths, the six-bit maximum, and the super-block
  payload sizes.

## 0.11.0 — 2026-11-27

The 8-bit float formats used for fp8 inference.

- `E4M3` (`float8_e4m3fn`): one sign, four exponent bits with a bias of 7,
  three significand bits, no infinity. `0x7E` is 448 and `0x7F` is NaN, so
  the NaN test is `(bits & 0x7F) == 0x7F` and not "exponent all ones"; the
  rules are stated in the source so the corner is not discovered by surprise.
- `E5M2` (`float8_e5m2`): five exponent bits, two significand bits,
  infinities at `0x7C`/`0xFC`, and six NaN patterns.
- Both widen exactly to binary32, narrow with round-to-nearest ties-to-even
  and gradual underflow, and carry the classification predicates. `E4M3`
  saturates at 448 as ONNX `Cast` does, with `from_float_overflow_nan` for
  callers that want overflow to become NaN; anything past the 480 midpoint
  takes one of those two paths. `E5M2` overflows to infinity.
- `fp8_vectors.mbt`: 256 rows per format, decoded with exact rational
  arithmetic by `tools/gen_fp8_vectors.py`. The tests compare every pattern,
  round-trip every finite one, check that the positive half increases
  strictly with the pattern, and count the NaNs (2 and 6).
- CLI: `fp8 <e4m3|e5m2> <bits>`.

## 0.10.0 — 2026-11-27

Two more GGML block formats, so a container's common tensor types can all be
read.

- `dequantize_tensor` now handles `Q5_0` (kind 6) and `Q8_0` (kind 8) next to
  `F32` and `Q4_0`. Q5_0 blocks are 22 bytes: one f16 scale, four bytes of
  fifth bits, sixteen packed nibbles, weight j taking bit j and weight j+16
  taking bit j+16. Q8_0 blocks are 34 bytes: one f16 scale and thirty-two
  signed bytes. `byte_size` follows the same three layouts.
- The fixture container grew from 384 to 576 bytes and now holds one block of
  each of Q4_0, Q5_0 and Q8_0 plus the F32 vector. The block readers are
  checked against a second implementation of the layouts written for
  `tools/emit_gguf_test.py`, which reads the fixture and emits the expected
  values as literals: every weight is a multiple of the 0.5 scale, so the
  comparison is exact and the assertion carries no tolerance.
- Docs and roadmap follow; the submission text still fits its byte limit.

## 0.9.0 — 2026-11-27

Arithmetic on the 16-bit float types, and the reader checked against a
released model.

- `F16::add / sub / mul / div` and `BF16::add / sub / mul / div`. One integer
  core serves both formats: the operands are decoded to `m * 2^e`, combined
  exactly in 64 bits, and rounded once when encoded. Round to nearest ties to
  even, gradual underflow, overflow to infinity, NaN and infinity
  propagation, and `(-0) + (-0)` keeping the sign. No binary64 instruction is
  used, so the results do not depend on backend float support.
- The bfloat16 exponent gap (2^-133 to 2^128) is wider than 64 bits can
  align. Beyond a fifty-bit gap the smaller operand is below half an ulp and
  the sum returns the larger operand unchanged; the sweep found this case by
  producing a zero from a shift of 126 bits, and a zero operand added to
  `1.0` then failed.
- `fp_vectors.mbt`: 171 operand pairs per format, generated by
  `tools/gen_fp_vectors.py` from exact rational arithmetic with a single
  rounding. Covers overflow edges, the subnormal floor, non-terminating
  quotients and a quotient landing exactly on a tie.
- `parse_gguf_header` reads only the fixed 24-byte prefix, so a file can be
  identified and sized without holding its payload.
- Tests against a released model: the first bytes of `stories260K.gguf`
  (sha256 `270cba1b…`, 48 tensors, 19 metadata entries) are in the suite, and
  the reader agrees with llama.cpp's reader and with an independent Python
  reader written for the test. A 384-byte container with a Q4_0 block and an
  F32 vector is parsed and dequantized end to end.
- Sixty-four tests, passing on `wasm`, `wasm-gc` and `js`.
- `tools/make_gguf_fixtures.py` and `tools/emit_gguf_test.py` rebuild the
  fixtures and the test from the files, so the embedded hex cannot drift.

## 0.8.1 — 2026-11-26

- `parse_gguf` checks every tensor extent against the buffer. A header that
  advertises payload which is not there now returns `None` instead of
  letting a later read run past the end.
- Test for a container truncated by eight payload bytes: the header still
  parses, so only the extent check catches it. Forty-three tests.
- `tools/make_gguf.py` regenerates the 324-byte fixture and prints the same
  hex the tests embed, byte for byte.

## 0.8.0 — 2026-11-26

GGUF container reading, so the types meet real data.

- `parse_gguf` reads a GGUF v3 header: version, alignment, and the tensor
  table with names, dimensions, ggml type and payload offset. Metadata is
  skipped except `general.alignment`; arrays recurse.
- `dequantize_tensor` turns `F32` and `Q4_0` tensors into binary64 values.
  Q4_0 goes block by block: 32 weights from 16 packed bytes, weight `j` in
  the low nibble and weight `j+16` in the high nibble.
- `bytes_from_hex` lets a container travel as text, because the buildable
  targets have no file access.
- `cmd/main gguf <hex>` runs the whole chain and prints per-tensor size plus
  the SNR of a further int8 quantization.
- Three tests against a 324-byte container built by an independent script:
  header fields, both tensor payloads, and rejection of a non-GGUF buffer.
  Forty-two tests total, passing on the three backends.

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
