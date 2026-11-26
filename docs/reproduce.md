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
`Total tests: 31, passed: 31, failed: 0`.

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
