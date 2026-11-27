# Reproduce the results

Requires `moon` (tested with 0.1.20261023 and 0.1.20261104) and a checkout of
this repository. Python 3 with `numpy` is needed only to regenerate the
reference tables, not to run the tests.

```sh
moon check
moon test --target wasm
moon test --target wasm-gc
moon test --target js
```

Expected: `moon check` finishes with no errors; on 0.1.20261023 no warnings,
on 0.1.20261104 the only output is the deprecation notice for
`Double::from_int` / `Int64::from_int`. Each test run prints
`Total tests: 76, passed: 76, failed: 0`. The command-line inspector
reproduces the numbers quoted in the proposal:

```sh
moon run cmd/main --target wasm -- f16 0x0002
moon run cmd/main --target wasm -- bf16 0x3F80
moon run cmd/main --target wasm -- fixed 3 1 16
moon run cmd/main --target wasm -- quant 1.25 0.05 0
moon run cmd/main --target wasm -- requant 25 0.05 0 0.01 128
moon run cmd/main --target wasm -- report
moon run cmd/main --target wasm -- verify
moon run cmd/main --target wasm -- selftest
```

The container command needs the file bytes on the command line, because
the targets this project builds for have no file access:

```sh
python tools/make_gguf.py            # writes tiny.gguf, prints its hex
moon run cmd/main --target wasm -- gguf <that hex>
```

`python tools/make_gguf.py` is deterministic: it writes 324 bytes, and the
same hex string is the one embedded in `gguf_wbtest.mbt`, so the fixture
used by the tests can be regenerated and compared byte for byte.

The arithmetic reference table and the GGUF fixtures are regenerated the same
way, and the generator refuses to guess:

```sh
python3 tools/gen_fp_vectors.py > fp_vectors.mbt
python3 tools/gen_fp8_vectors.py > fp8_vectors.mbt
python3 tools/make_gguf_fixtures.py path/to/stories260K.gguf   # prints hex
python3 tools/emit_gguf_test.py path/to/stories260K.gguf       # rewrites the test
```

`gen_fp_vectors.py` prints the table on stdout and writes the numpy cross-check
to stderr: `numpy float16: 684 agreement, 0 disagreement`. The model used for
the real-file check is `stories260K.gguf`, 1,185,376 bytes, sha256
`270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d`, available
from `ggml-org/models` on Hugging Face. It is not vendored here; only its
first 24 bytes are, because that is what the identification path reads.

`f16 0x0002` prints `value=1.1920929e-7` (2 × 2^-24), `requant` prints
`multiplier=1342177280 shift=28 code=253 value=1.25`, `report` prints
49.64783777764896 dB for nearest-even against 43.7 to 44.3 dB for the three
truncating modes, `verify` prints `vectors 60/60 reproduce`, and `selftest`
prints `selftest 7/7 passed`.

## What each exhaustive test covers

- `exhaustive roundtrip over all 65536 patterns` — all binary16 patterns
  except NaN widen to binary32 and narrow back unchanged (63490 checked).
- `exhaustive roundtrip over all 65536 bfloat16 patterns` — 65282 checked.
- `f16 add over all patterns against the probes` — 65536 patterns × 12
  probes, 786432 results against the double path.
- `f16 multiply over all patterns against the probes` — another 786432.
- `f16 divide and subtract over all patterns against the probes` — 1572864.
- `bf16 add and multiply over all patterns against the probes` — 1572864.
- `bf16 divide and subtract over all patterns against the probes` — 1572864.
- `binary16 arithmetic reproduces the exact reference table` — 171 pairs ×
  four operations, values from rational arithmetic, 684 bit-exact results;
  the same for bfloat16. `the reference table covers the boundaries` asserts
  the table really reaches subnormals and infinities.
- `exhaustive 8-bit multiplication across five rounding modes` — 65536
  operand pairs × 5 modes compared with an independent `Double` reference.
- `exhaustive widening multiply is exact` — the same pairs, raw product
  returned bit for bit.
- `exhaustive saturate and wrap on every 16-bit raw value` — all 65536 raw
  values, adding one, both overflow modes.
- `integral requantization matches the double reference` — all 256 int8
  codes.
- `nearest even never exceeds half a step` — 990 signal points across
  -0.99..0.99, bound is half the scale; `toward zero is worse than nearest
  even` requires a strictly larger maximum error on the same points.
- `snr matches a double reference` — the reported decibels equal
  `10 * log10(signal / noise)` computed directly in `Double`.
- `published vectors reproduce` — all 60 published rows re-checked against
  the live implementation, and `vector table covers the extremes of every
  type` pins both ends of each type so a trimmed table cannot pass silently.
- `parse the tiny gguf header` — every header field of the 324-byte
  container: version, alignment, both tensor names, dimensions, ggml types,
  offsets and computed byte sizes.
- `dequantize a q4_0 tensor and an f32 tensor` — 64 weights block by block,
  both nibble orders, plus the eight F32 values.
- `the fixed header of a released model is read correctly` — the first 24
  bytes of `stories260K.gguf` resolve to version 3, 48 tensors and 19
  metadata entries, the same values llama.cpp reports.
- `a short buffer is enough to identify a file` — and the same bytes with a
  broken magic are refused.
- `a fixture container parses and dequantizes` — a 1264-byte container with
  one block each of Q4_0, Q5_0, Q8_0, Q4_K, Q6_K and Q5_K plus an F32 vector;
  offsets, payload position and thirty-five sampled weights end to end.
- `k4 scale and min unpacking covers both bit paths` — the twelve packed
  bytes of a K-quant super-block against an eight-row table written by hand.
- `k4 scale and min reach the six bit maximum` — 63 in the low six bits and
  the nibble-only path for the second four.
- `K-quant payload sizes follow the 256-weight super-block` — 144, 210 and
  176 bytes per block, and their double-block multiples.
- `a non-gguf byte array is rejected` — 24 bytes of junk must not parse.
- `a truncated payload is rejected` — the same header with the last eight
  payload bytes removed: the header alone still parses, so only the extent
  check can catch it.
- `every e4m3 pattern agrees with the rational table` — all 256 patterns,
  value by value against `fp8_vectors.mbt`.
- `every e5m2 pattern agrees with the rational table` — the same for the
  other format, including its infinities.
- `narrowing reproduces every finite pattern` — both formats, signed zero and
  subnormals included.
- `positive values increase with the pattern` — strict monotonicity on the
  positive half of each format.
- `e4m3 saturates or reports overflow, and has no infinity` and
  `e5m2 keeps ieee infinities` — the 480 rounding boundary, an infinity input
  that E4M3 cannot store, and the infinities E5M2 can.
- `e4m3 subnormal rounding and the tie at 1.5 ulp` — the subnormal floor and
  a tie that ties-to-even must break toward the even pattern.
- `nan patterns are exactly the ones the formats define` — 2 and 6.
- `published constants match their patterns` — 448, 57344, 2^-9, 2^-6, 2^-16,
  2^-14.

## Where the reference values come from

The float conversions are checked by their own round-trip and monotonicity
properties over the full pattern space, which is stronger than comparing a
handful of values against a published table.

The arithmetic reference table is built from `fractions.Fraction`: each
operand is decoded to an exact rational, the operation is carried out exactly,
and the result is rounded once by the rules of IEEE 754-2019 — no floating
point anywhere on the reference side, so a double-rounding habit cannot be
inherited. As a third opinion the same binary16 pairs are also evaluated with
numpy's `float16`, which agreed on all 684 results.

Fixed-point and quantization results are compared against an independently
written `Double` path, so a mistake would have to be made twice in exactly the
same way to pass.

The fp8 table is produced by `tools/gen_fp8_vectors.py` from `Fraction`
arithmetic written straight from the two format definitions — no float
anywhere — so every row is exact and the MoonBit side is compared against
values it had no part in producing.

The GGUF fixtures come from `tools/make_gguf_fixtures.py`, which is a separate
implementation of the container format from the one under test. The block
layouts are checked a second way: `tools/emit_gguf_test.py` reads the fixture
and dequantizes it with an implementation written from the layouts alone,
then writes the expected values into the test as literals, so the two sides
share no code. The real file's header values were cross-checked with
llama.cpp's reader.

A third opinion covers the K-quant blocks. Installing the `gguf` package
(`python -m pip install gguf`) and dequantizing the same block bytes through
`gguf.quants.dequantize` gives values that match the fixture literals exactly,
with a maximum difference of 0.0 on all 256 values of each of the three
blocks. That package is maintained by the organisation that publishes
llama.cpp, so the layout is not being judged by this repository alone —
`soehi/gguf` 0.1.3, an independent MoonBit module, reads the same `Q4_K` and
`Q6_K` blocks too.

## Known platform constraints

`Int` is a signed 32-bit word here. Fixed-point multiplication therefore
supports fields up to 16 bits (product below 2^30), and requantization
forms its ~2^38 product in `Int64`. Both limits are in the README and in
the changelog.
