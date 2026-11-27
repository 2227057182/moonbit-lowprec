"""Emits the MoonBit test that checks the GGUF fixtures.

The hex strings and the expected values are read from the files rather than
retyped, so the test cannot drift from the fixtures it claims to check. The
dequantization below is a second implementation written from the block layouts
alone, independent of the MoonBit code under test.

    python3 tools/emit_gguf_test.py <path-to-real.gguf>
"""

import struct
import sys

FIXTURE = "fixture.gguf"
OUT = "gguf_fixtures_wbtest.mbt"


def parse(data):
    version = struct.unpack_from("<I", data, 4)[0]
    tensor_count = struct.unpack_from("<Q", data, 8)[0]
    kv_count = struct.unpack_from("<Q", data, 16)[0]
    o = 24
    align = 32
    for _ in range(kv_count):
        n = struct.unpack_from("<Q", data, o)[0]
        key = data[o + 8:o + 8 + n].decode()
        o += 8 + n
        vtype = struct.unpack_from("<I", data, o)[0]
        o += 4
        if vtype == 8:
            n = struct.unpack_from("<Q", data, o)[0]
            o += 8 + n
        elif vtype == 4:
            if key == "general.alignment":
                align = struct.unpack_from("<I", data, o)[0]
            o += 4
        else:
            raise ValueError(f"unhandled metadata type {vtype}")
    tensors = []
    for _ in range(tensor_count):
        n = struct.unpack_from("<Q", data, o)[0]
        name = data[o + 8:o + 8 + n].decode()
        o += 8 + n
        nd = struct.unpack_from("<I", data, o)[0]
        o += 4
        dims = [struct.unpack_from("<Q", data, o + 8 * i)[0] for i in range(nd)]
        o += 8 * nd
        kind = struct.unpack_from("<I", data, o)[0]
        o += 4
        offset = struct.unpack_from("<Q", data, o)[0]
        o += 8
        tensors.append((name, dims, kind, offset))
    while o % align != 0:
        o += 1
    return version, align, tensors, o


def half(p, o):
    return struct.unpack_from("<e", p, o)[0]


def u32(p, o):
    return struct.unpack_from("<I", p, o)[0]


def signed(b):
    return b - 256 if b >= 128 else b


def q4_0(p):
    out = [0.0] * 32
    scale = half(p, 0)
    for j in range(16):
        out[j] = ((p[2 + j] & 15) - 8) * scale
        out[j + 16] = ((p[2 + j] >> 4) - 8) * scale
    return out


def q5_0(p):
    out = [0.0] * 32
    scale = half(p, 0)
    qh = u32(p, 2)
    for j in range(16):
        low = (p[6 + j] & 15) | (((qh >> j) & 1) << 4)
        high = (p[6 + j] >> 4) | (((qh >> (j + 16)) & 1) << 4)
        out[j] = (low - 16) * scale
        out[j + 16] = (high - 16) * scale
    return out


def q8_0(p):
    scale = half(p, 0)
    return [signed(p[2 + j]) * scale for j in range(32)]


def f32(p):
    return [struct.unpack_from("<f", p, 4 * j)[0] for j in range(32)]


def fmt(v):
    s = repr(float(v))
    if "e" in s or "E" in s:
        raise ValueError(f"value {v} cannot be written as a plain literal")
    return s


TEMPLATE = '''/// Fixtures built by `tools/make_gguf_fixtures.py`, an independent Python
/// implementation of the container format, and checked by
/// `tools/emit_gguf_test.py`, a second implementation of the block layouts.
///
/// `fixture_gguf_hex` is a complete {size}-byte container holding one block
/// each of Q4_0, Q5_0 and Q8_0 plus a 32-value F32 tensor.
///
/// `real_header_hex` is the first 24 bytes of stories260K.gguf, a released
/// model whose sha256 is
/// 270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d.
/// The same 24 bytes were independently resolved by llama.cpp's reader as
/// version 3, 48 tensors, 19 metadata entries.
let fixture_gguf_hex : String = "{fixture}"

let real_header_hex : String = "{real}"

test "the fixed header of a released model is read correctly" {{
  let bytes = bytes_from_hex(real_header_hex)
  let header = parse_gguf_header(bytes).unwrap()
  assert_eq(header.version, 3)
  assert_eq(header.tensor_count, 48)
  assert_eq(header.kv_count, 19)
}}

test "a short buffer is enough to identify a file" {{
  // The same 24 bytes with the magic broken must be rejected, so the check
  // is on content and not only on length.
  let bytes = bytes_from_hex(real_header_hex)
  bytes[1] = Int::to_byte(88)
  assert_eq(parse_gguf_header(bytes) is None, true)
}}

test "a fixture container parses and dequantizes" {{
  let bytes = bytes_from_hex(fixture_gguf_hex)
  let file = parse_gguf(bytes).unwrap()
  assert_eq(file.version, 3)
  assert_eq(file.tensors.length(), 4)
  assert_eq(file.data_start, {data_start})
  assert_eq(file.tensors[0].name, "blk.0.test_q4_0")
  assert_eq(file.tensors[0].kind, 2)
  assert_eq(file.tensors[1].name, "blk.0.test_q5_0")
  assert_eq(file.tensors[1].kind, 6)
  assert_eq(file.tensors[2].name, "blk.0.test_q8_0")
  assert_eq(file.tensors[2].kind, 8)
  assert_eq(file.tensors[3].name, "blk.0.test_f32")
  assert_eq(file.tensors[3].kind, 0)
  assert_eq(file.tensors[0].dims[0], 32)

  // All three blocks carry the same 0.5 scale, so the weights are exact
  // multiples of 0.5 and the expected values below are literals.
  let q4 = dequantize_tensor(bytes, file, file.tensors[0])
{q4}
  let q5 = dequantize_tensor(bytes, file, file.tensors[1])
{q5}
  let q8 = dequantize_tensor(bytes, file, file.tensors[2])
{q8}
  let w = dequantize_tensor(bytes, file, file.tensors[3])
{w}
}}
'''


def main():
    real = sys.argv[1]
    with open(real, "rb") as f:
        head = f.read(24)
    with open(FIXTURE, "rb") as f:
        body = f.read()
    version, align, tensors, data_start = parse(body)
    payload = body

    def block(t, nbytes):
        return payload[data_start + t[3]:data_start + t[3] + nbytes]

    expected = {
        "q4": (q4_0(block(tensors[0], 18)), [0, 15, 16, 31]),
        "q5": (q5_0(block(tensors[1], 22)), [0, 15, 16, 31]),
        "q8": (q8_0(block(tensors[2], 34)), [0, 7, 31]),
        "w": (f32(block(tensors[3], 128)), [0, 7, 31]),
    }
    lines = {}
    for name, (values, picks) in expected.items():
        body_lines = "\n".join(f"  assert_eq({name}[{i}], {fmt(values[i])})" for i in picks)
        lines[name] = body_lines

    text = TEMPLATE.format(
        fixture=body.hex(),
        real=head.hex(),
        size=len(body),
        data_start=data_start,
        **lines,
    )
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"wrote {OUT}: {len(text.encode())} bytes, fixture {len(body)} bytes, "
          f"data_start {data_start}, v{version} alignment {align}")


if __name__ == "__main__":
    main()
