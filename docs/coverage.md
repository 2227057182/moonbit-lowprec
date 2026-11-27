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
| `int8` | 0 | no integer quantization type in any description |
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
- **`matrixlt/moon-onnx` 0.2.0** — pure-MoonBit ONNX inference engine.
  Re-checked at source level rather than from the description: the repository
  is `matrixlt/moon-onnx`, and `packages/ml_kernels/quant/quant.mbt` implements
  `quantize_u8_per_tensor`, `quantize_i8_per_tensor`, `dequantize_u8_per_tensor`,
  `dequantize_i8_per_tensor` and `round_half_to_even`, following the
  `QuantizeLinear` / `QLinearMatMul` specification.
  `packages/onnx/ops/quant.mbt` wraps them as graph operators.

  What that means for the gap claim, stated plainly: **per-tensor int8/uint8
  affine quantization is not a gap by itself.** Those operators exist and are
  written correctly (round half to even, clamped to the integer range).

  What is still absent, checked by code search over that repository:
  `bfloat16`, `float16`, `per_channel`, `requant` all return 0 matches. So
  there is no 16-bit float type, no per-channel scale, no integer-only
  requantization path, no fixed-point type, no GGML block format (Q4_0, Q5_0,
  Q8_0), and no error reporting. The operator lives inside a graph runner and
  is not an importable type layer.

  This project's contribution should therefore be argued on those axes, not
  on "no int8 support in the ecosystem".

## 16-bit arithmetic

`moonbit-community/softfloat` does implement software floating-point
arithmetic with explicit rounding, but over its own emulated types rather than
as binary16 or bfloat16 values a caller can store and combine. `matrixlt/moon-onnx`
has no `float16` or `bfloat16` anywhere, so it has no arithmetic on them
either. Nothing in the index exposes "a binary16 value you can add, subtract,
multiply and divide with a correctly rounded result" — and a type that can be
stored but not operated on is a conversion table, not a numeric layer. That is
the axis this release fills, which is also why the work is done on the
significand instead of by widening to binary64.

## 8-bit float formats

The same scan returns no `float8`, `e4m3` or `e5m2` match. `matrixlt/moon-onnx`
casts tensors through ONNX's `float8` types but does not expose an 8-bit float
as a store-and-inspect type, and `soehi/gguf`'s block readers decode 4- and
5-bit integers rather than 8-bit floats. What is missing is not the arithmetic
(the formats are conversion and storage formats, not compute formats) but the
exact decode, the classification predicates, and the rounding behaviour on the
way back — including E4M3's unusual corner where the largest finite value and
the NaN share an all-ones exponent.

## GGUF block coverage in the ecosystem

`soehi/gguf` 0.1.3 is more than a header reader. Its `src/tensor/dequant.mbt`
dispatches on ggml type ids 0, 1, 2, 6, 8, 12 and 14 — F32, F16, Q4_0, Q5_0,
Q8_0, Q4_K and Q6_K — and its `scale_min_k4` unpacks the same twelve bytes this
project does. Stated plainly: **Q4_K and Q6_K dequantization is not a gap**,
because a MoonBit caller can already get those weights as `Float`.

What that module does not provide, checked file by file: no `Q5_K` (id 13),
no `Q2_K` (10) or `Q3_K` (11), no binary16 or bfloat16 arithmetic, no fp8, no
fixed-point format with a selectable rounding mode, and no error report. Its
output is `Float` tied to the `Bytes` of a GGUF payload. The layer this project
argues for — a value you can store, convert with a stated rounding, and feed
into the same report as every other format — sits outside it. That is the axis,
not the block map.

As of 0.14.0 the five K-quant types are complete here — 10, 11, 12, 13 and 14 —
while the module above stops at 12 and 14. The two new ones are the smallest
prize in this release and the honest framing is that completeness is the claim,
not novelty: `Q2_K` and `Q3_K` were the only K-quant formats a caller could not
read in MoonBit at all.

## Writing GGUF containers in the ecosystem

The same file-by-file check turns up nothing on the writing side. `soehi/gguf`
exposes dequantization functions — payload bytes in, values out — and no
function that produces bytes: no header assembly, no tensor entry, no
quantizer. `xfp8` and `msl-gguf` likewise stop at values. Serializing a
container in MoonBit was, as far as this scan reaches, not done.

That absence is worth one sentence of caution, because "nobody does it" is
the kind of claim this project has been told not to lean on. The claim here
is narrower and checkable: the bytes `build_gguf` writes are opened by `gguf`,
the Python package maintained by the organisation that publishes llama.cpp,
which reports both tensors, their types and shapes, and dequantizes them with
maximum errors of exactly half a quantization step. A writer that another
implementation accepts is not a self-argument.

What is not written, stated rather than implied: metadata pairs (`kv_count`
is zero), and the K-quant super-block layouts. Only `Q8_0` and `Q4_0` have
quantizers, because those are the two blocks whose packing the reader already
takes apart here and whose bytes the fixture round trip could be checked
against.

## Conversions between the low-precision formats

The 3975-module scan found no module that converts between binary16 and
bfloat16 in MoonBit at all, let alone between the two 8-bit float formats;
the patterns such a module would need (xfp8, msl-gguf, and the K-quant reader
above) stop at producing a `Float` and never take a low-precision pattern as
input again. So the six directions here are new surface, not a rebuild.

The claim they carry is narrower and checkable: one rounding, on the
significand, with no binary32 step in between. Two of the six do not round at
all, and the changelog says which and why rather than calling all conversions
"exact".

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
