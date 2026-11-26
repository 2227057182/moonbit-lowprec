# Reproduce the results

Requires `moon` (tested with 0.1.20261023) and a checkout of this
repository.

```sh
moon check
moon test --target wasm
moon test --target wasm-gc
moon test --target js
```

Expected: `moon check` finishes with no warnings; each test run prints
`Total tests: 43, passed: 43, failed: 0`. The command-line inspector
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

`f16 0x0002` prints `value=1.1920929e-7` (2 × 2^-24), `requant` prints
`multiplier=1342177280 shift=28 code=253 value=1.25`, `report` prints
49.64783777764896 dB for nearest-even against 43.7 to 44.3 dB for the three
truncating modes, `verify` prints `vectors 60/60 reproduce`, and `selftest`
prints `selftest 7/7 passed`.

## What each exhaustive test covers

- `exhaustive roundtrip over all 65536 patterns` — all binary16 patterns
  except NaN widen to binary32 and narrow back unchanged (63490 checked).
- `exhaustive roundtrip over all 65536 bfloat16 patterns` — 65282 checked.
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
- `a non-gguf byte array is rejected` — 24 bytes of junk must not parse.
- `a truncated payload is rejected` — the same header with the last eight
  payload bytes removed: the header alone still parses, so only the extent
  check can catch it.

## Where the reference values come from

No value is copied from an external tool. The float conversions are checked
by their own round-trip and monotonicity properties over the full pattern
space, which is stronger than comparing a handful of values against a
published table. Fixed-point and quantization results are compared against
an independently written `Double` path, so a mistake would have to be made
twice in exactly the same way to pass.

## Known platform constraints

`Int` is a signed 32-bit word here. Fixed-point multiplication therefore
supports fields up to 16 bits (product below 2^30), and requantization
forms its ~2^38 product in `Int64`. Both limits are in the README and in
the changelog.
