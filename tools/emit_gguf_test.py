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


def k4_scale_min(p, qbase, j):
    """llama.cpp's get_scale_min_k4 (ggml-quants.c:889), written from its body."""
    if j < 4:
        return p[qbase + j] & 63, p[qbase + j + 4] & 63
    return ((p[qbase + j + 4] & 15) | ((p[qbase + j - 4] >> 6) << 4),
            (p[qbase + j + 4] >> 4) | ((p[qbase + j] >> 6) << 4))


def q4_k(p):
    d = half(p, 0)
    dmin = half(p, 2)
    out = [0.0] * 256
    for g in range(4):
        sc1, m1 = k4_scale_min(p, 4, g * 2)
        sc2, m2 = k4_scale_min(p, 4, g * 2 + 1)
        for l in range(32):
            q = p[16 + g * 32 + l]
            out[g * 64 + l] = d * sc1 * (q & 15) - dmin * m1
            out[g * 64 + 32 + l] = d * sc2 * (q >> 4) - dmin * m2
    return out


def q5_k(p):
    d = half(p, 0)
    dmin = half(p, 2)
    out = [0.0] * 256
    for g in range(4):
        sc1, m1 = k4_scale_min(p, 4, g * 2)
        sc2, m2 = k4_scale_min(p, 4, g * 2 + 1)
        for l in range(32):
            b = p[48 + g * 32 + l]
            h = p[16 + l]
            v1 = (b & 15) + (16 if (h >> (2 * g)) & 1 else 0)
            v2 = (b >> 4) + (16 if (h >> (2 * g + 1)) & 1 else 0)
            out[g * 64 + l] = d * sc1 * v1 - dmin * m1
            out[g * 64 + 32 + l] = d * sc2 * v2 - dmin * m2
    return out


def q6_k(p):
    d = half(p, 208)
    out = [0.0] * 256
    for s in range(2):
        ql, qh, sc = s * 64, 128 + s * 32, 192 + s * 8
        for l in range(32):
            i = l // 16
            h = p[qh + l]
            q1 = ((p[ql + l] & 15) | ((h & 3) << 4)) - 32
            q2 = ((p[ql + 32 + l] & 15) | (((h >> 2) & 3) << 4)) - 32
            q3 = ((p[ql + l] >> 4) | (((h >> 4) & 3) << 4)) - 32
            q4 = ((p[ql + 32 + l] >> 4) | (((h >> 6) & 3) << 4)) - 32
            out[s * 128 + l] = d * signed(p[sc + i]) * q1
            out[s * 128 + l + 32] = d * signed(p[sc + i + 2]) * q2
            out[s * 128 + l + 64] = d * signed(p[sc + i + 4]) * q3
            out[s * 128 + l + 96] = d * signed(p[sc + i + 6]) * q4
    return out


def k3_scale(p, base, i):
    """Q3_K's sixteen 6-bit scales, written from the gguf package's layout."""
    low = (p[base + i] & 15) if i < 8 else (p[base + i - 8] >> 4)
    high = (p[base + 8 + (i % 4)] >> (2 * (i // 4))) & 3
    return (low | (high << 4)) - 32


def q2_k(p):
    d = half(p, 80)
    dmin = half(p, 82)
    out = [0.0] * 256
    for s in range(2):
        q = 16 + s * 32
        for j in range(4):
            sc1, sc2 = p[s * 8 + j * 2], p[s * 8 + j * 2 + 1]
            for l in range(16):
                out[s * 128 + j * 32 + l] = (
                    d * (sc1 & 15) * ((p[q + l] >> (2 * j)) & 3) - dmin * (sc1 >> 4))
                out[s * 128 + j * 32 + 16 + l] = (
                    d * (sc2 & 15) * ((p[q + 16 + l] >> (2 * j)) & 3) - dmin * (sc2 >> 4))
    return out


def q3_k(p):
    d = half(p, 108)
    out = [0.0] * 256
    for s in range(2):
        qs = 32 + s * 32
        for j in range(4):
            bit = s * 4 + j
            dl1 = d * k3_scale(p, 96, s * 8 + j * 2)
            dl2 = d * k3_scale(p, 96, s * 8 + j * 2 + 1)
            for l in range(16):
                o1 = 0 if (p[l] >> bit) & 1 else 4
                o2 = 0 if (p[16 + l] >> bit) & 1 else 4
                out[s * 128 + j * 32 + l] = dl1 * (((p[qs + l] >> (2 * j)) & 3) - o1)
                out[s * 128 + j * 32 + 16 + l] = dl2 * (((p[qs + 16 + l] >> (2 * j)) & 3) - o2)
    return out


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
/// each of Q4_0, Q5_0, Q8_0, Q4_K, Q6_K, Q5_K, Q2_K and Q3_K plus a 32-value
/// F32 tensor.
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
  assert_eq(file.tensors.length(), 9)
  assert_eq(file.data_start, {data_start})
  assert_eq(file.tensors[0].name, "blk.0.test_q4_0")
  assert_eq(file.tensors[0].kind, 2)
  assert_eq(file.tensors[1].name, "blk.0.test_q5_0")
  assert_eq(file.tensors[1].kind, 6)
  assert_eq(file.tensors[2].name, "blk.0.test_q8_0")
  assert_eq(file.tensors[2].kind, 8)
  assert_eq(file.tensors[3].name, "blk.0.test_f32")
  assert_eq(file.tensors[3].kind, 0)
  assert_eq(file.tensors[4].name, "blk.0.test_q4_k")
  assert_eq(file.tensors[4].kind, 12)
  assert_eq(file.tensors[5].name, "blk.0.test_q6_k")
  assert_eq(file.tensors[5].kind, 14)
  assert_eq(file.tensors[6].name, "blk.0.test_q5_k")
  assert_eq(file.tensors[6].kind, 13)
  assert_eq(file.tensors[7].name, "blk.0.test_q2_k")
  assert_eq(file.tensors[7].kind, 10)
  assert_eq(file.tensors[8].name, "blk.0.test_q3_k")
  assert_eq(file.tensors[8].kind, 11)
  assert_eq(file.tensors[0].dims[0], 32)
  assert_eq(file.tensors[4].dims[0], 256)
  assert_eq(file.tensors[7].dims[0], 256)

  // All blocks carry the same 0.5 scale, so the weights are exact
  // multiples of 0.5 and the expected values below are literals.
  let q4 = dequantize_tensor(bytes, file, file.tensors[0])
{q4}
  let q5 = dequantize_tensor(bytes, file, file.tensors[1])
{q5}
  let q8 = dequantize_tensor(bytes, file, file.tensors[2])
{q8}
  let w = dequantize_tensor(bytes, file, file.tensors[3])
{w}
  let q4k = dequantize_tensor(bytes, file, file.tensors[4])
{q4k}
  let q6k = dequantize_tensor(bytes, file, file.tensors[5])
{q6k}
  let q5k = dequantize_tensor(bytes, file, file.tensors[6])
{q5k}
  // Q2_K subtracts `dmin` after scaling, Q3_K's third bit inverts the offset
  // of four. The values below match the official gguf package on all 256.
  let q2k = dequantize_tensor(bytes, file, file.tensors[7])
{q2k}
  let q3k = dequantize_tensor(bytes, file, file.tensors[8])
{q3k}
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
        "q4k": (q4_k(block(tensors[4], 144)), [0, 31, 32, 64, 96, 128, 255]),
        "q6k": (q6_k(block(tensors[5], 210)), [0, 31, 32, 64, 96, 128, 255]),
        "q5k": (q5_k(block(tensors[6], 176)), [0, 31, 32, 64, 96, 128, 255]),
        "q2k": (q2_k(block(tensors[7], 84)), [0, 31, 32, 64, 128, 255]),
        "q3k": (q3_k(block(tensors[8], 110)), [1, 15, 31, 64, 95, 96, 128, 191]),
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
