# Coverage check: what the ecosystem already has

Snapshot taken 2026-11-26 from `https://mooncakes.io/api/v0/modules`, which
returns the full module index in one response (3975 modules). Counts are
substring matches over `name + description + keywords`, then verified by
reading the matching descriptions, because plain substring matching
produces false positives.

## Low-precision numeric layer

| Query | Matches | Meaning |
|---|---|---|
| `fixed point` | 0 | no fixed-point arithmetic |
| `int8` | 0 | no integer quantization type |
| `int16` | 0 | no 16-bit integer type |
| `f16` / `fp16` / `half float` | 0 | no binary16 |
| `bfloat16` | 0 | no bfloat16 |
| `float16` | 0 | no float16 |
| `correctly rounded` | 0 | no rounding-correctness suite |

Noise worth naming: `quantiz` has 6 matches, all image colour quantization
(`pixelate`) or GGUF container reading (`soehi/gguf`, "read-only GGUF
container parsing and ggml block dequantization"). `saturat` has 3 matches,
all equality saturation in e-graph libraries — a different meaning of the
word.

## Adjacent modules and why they are not the same thing

- **`Luna-Flow/floating` 0.12.0-pre** — "Arbitrary-precision binary, decimal,
  and interval arithmetic ... with checked operations". Different numeric
  domain: arbitrary precision and intervals, not 16-bit storage formats.
- **`moonbit-community/softfloat` 0.1.0** — "Software floating-point
  arithmetic with explicit rounding and exception state". Emulates IEEE
  float semantics without providing binary16 or bfloat16 types.
- **`soehi/gguf` 0.1.3** — reads GGUF headers and dequantizes ggml blocks
  into float. It consumes low-precision data; it does not expose the types.
  This is the strongest evidence that demand exists while the type layer
  does not.
- **`matrixlt/onnx` 0.2.0** — pure-MoonBit ONNX inference engine. If it
  implements quantization internally, that implementation is not an
  exposed, testable type layer; this entry should be re-checked before the
  quantization milestone.

## Fields where the ecosystem is already dense (do not rebuild)

Recorded so the choice above is a deliberate comparison, not an accident:

| Area | Existing modules |
|---|---|
| Unicode normalization / collation | `Atman-Angle/moonbit-unicode`, `CaptainK-65/moon-collate`, `chawyehsu/stringprep` |
| Linear programming | `Freon793/moonopt` (sparse revised simplex + presolve) |
| Graph algorithms | `moonbitstack/moongraph` plus at least seven pathfinding modules |
| Exact decimal / money | `moonbit-community/decimal`, `haol-05/moonmoney` |
| Robust geometric predicates | `t-ujiie-g/robust`, `ZYK-ai-nb/moon-geo` |
| Mutation testing | `f4ah6o/turtles`, `Magic486/moon_mutest` |
| Cross-target behaviour parity | `ZhangBowen2006/moonbit-target-parity`, `ZBZ-ai-nb/cakecheck` |
| License / supply-chain audit | `clbbbb/moonbit-license-audit`, `Delthin/moon_license_scan` |

## Method limits

The index has no server-side search; every `?q=`, `?search=`, `?page=`
parameter is ignored and the full list is returned. Filtering is therefore
local and substring based. A module whose description omits the relevant
term will not appear above, so the counts are a lower bound on density, and
the zero rows were confirmed by reading the descriptions of the closest
matches rather than trusted blindly.
