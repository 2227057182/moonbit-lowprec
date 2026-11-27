"""Builds the GGUF fixtures and the test that checks them.

Everything here is an independent second implementation of the container
format: a small reader and writer written straight from the GGUF v3 layout,
with no dependency on the MoonBit code under test and none on llama.cpp.

Two outputs:

1. The first 24 bytes of a real released model (stories260K.gguf, sha256
   270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d). The
   reader is therefore checked against a file this repository did not produce,
   and the values below were cross-checked with llama.cpp's own gguf reader.

2. A complete container with one block each of Q4_0, Q5_0 and Q8_0 plus a
   thirty-two value F32 tensor, written by this writer and read back by this
   reader, then embedded in the test.

    python3 tools/make_gguf_fixtures.py <path-to-real.gguf>
"""

import hashlib
import struct
import sys

MAGIC = b"GGUF"
ALIGNMENT = 32

# value type ids, as defined by the GGUF v3 header
T_STRING = 8
T_UINT32 = 4
T_BOOL = 7


def read_string(data, o):
    n = struct.unpack_from("<Q", data, o)[0]
    o += 8
    return data[o:o + n].decode(), o + n


def skip_value(data, o, vtype):
    if vtype in (0, 1):
        return o + 1
    if vtype in (2, 3):
        return o + 2
    if vtype in (4, 5, 6):
        return o + 4
    if vtype == 7:
        return o + 1
    if vtype == T_STRING:
        n = struct.unpack_from("<Q", data, o)[0]
        return o + 8 + n
    if vtype == 9:
        elem = struct.unpack_from("<I", data, o)[0]
        count = struct.unpack_from("<Q", data, o + 4)[0]
        o += 12
        for _ in range(count):
            o = skip_value(data, o, elem)
        return o
    if vtype in (10, 11, 12):
        return o + 8
    raise ValueError(f"unhandled value type {vtype}")


def read_tensor(data, o):
    name, o = read_string(data, o)
    n_dims = struct.unpack_from("<I", data, o)[0]
    o += 4
    dims = []
    for _ in range(n_dims):
        dims.append(struct.unpack_from("<Q", data, o)[0])
        o += 8
    kind = struct.unpack_from("<I", data, o)[0]
    o += 4
    offset = struct.unpack_from("<Q", data, o)[0]
    o += 8
    return {"name": name, "dims": dims, "kind": kind, "offset": offset}, o


def parse_container(data):
    if data[:4] != MAGIC:
        raise ValueError("not a GGUF file")
    version = struct.unpack_from("<I", data, 4)[0]
    tensor_count = struct.unpack_from("<Q", data, 8)[0]
    kv_count = struct.unpack_from("<Q", data, 16)[0]
    o = 24
    alignment = ALIGNMENT
    kv = {}
    for _ in range(kv_count):
        key, o = read_string(data, o)
        vtype = struct.unpack_from("<I", data, o)[0]
        o += 4
        if key == "general.alignment" and vtype == T_UINT32:
            alignment = struct.unpack_from("<I", data, o)[0]
        kv[key] = vtype
        o = skip_value(data, o, vtype)
    tensors = []
    for _ in range(tensor_count):
        t, o = read_tensor(data, o)
        tensors.append(t)
    data_start = o
    while data_start % alignment != 0:
        data_start += 1
    return version, alignment, kv, tensors, data_start


def q4_k_block():
    """One 144-byte Q4_K super-block of 256 weights.

    Fields, in order: f16 d, f16 dmin, twelve packed scale/min bytes, then
    128 bytes of 4-bit weights. The twelve bytes are laid out so that all
    eight sub-blocks differ and every unpack path is exercised: scales
    1..4 and mins 5..8 live in the low six bits, the high two bits of the
    first four bytes raise scales 4..7 to 17..20, and the nibbles of the
    last four bytes supply their low four bits.
    """
    head = struct.pack("<e", 0.5) + struct.pack("<e", 0.25)
    scales = bytes([0x41, 0x42, 0x43, 0x44, 5, 6, 7, 8, 0x11, 0x22, 0x33, 0x44])
    qs = bytes(((l + g) % 16) | (((l + g + 4) % 16) << 4) for g in range(4) for l in range(32))
    return head + scales + qs


def q5_k_block():
    """One 176-byte Q5_K super-block of 256 weights.

    Same twelve packed bytes as Q4_K, then 32 bytes of fifth bits and 128
    bytes of 4-bit weights. Group g reads bit 2g for the low nibbles and bit
    2g+1 for the high nibbles of the shared byte, so all eight bits of every
    byte are used across the four groups.
    """
    head = struct.pack("<e", 0.5) + struct.pack("<e", 0.25)
    scales = bytes([0x41, 0x42, 0x43, 0x44, 5, 6, 7, 8, 0x11, 0x22, 0x33, 0x44])
    qh = bytes((l * 3) & 0xFF for l in range(32))
    qs = bytes(((l + g) % 16) | (((l + g + 4) % 16) << 4) for g in range(4) for l in range(32))
    return head + scales + qh + qs


def q6_k_block():
    """One 210-byte Q6_K super-block of 256 weights: ql, qh, int8 scales, f16 d."""
    ql = bytes(((l + 3 * s) % 16) | (((l + 5 * s + 7) % 16) << 4)
               for s in range(2) for l in range(64))
    qh = bytes((l % 4) | (((l + 1) % 4) << 2) | (((l + 2) % 4) << 4) | (((l + 3) % 4) << 6)
               for s in range(2) for l in range(32))
    scales = struct.pack("<16b", -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7)
    return ql + qh + scales + struct.pack("<e", 0.25)


def q2_k_block():
    """One 84-byte Q2_K super-block of 256 weights.

    Sixteen scale bytes (low nibble times d, high nibble times dmin), then
    64 bytes of four 2-bit weights each, then f16 d and f16 dmin. The low
    nibbles walk 1..15 and the high nibbles walk 2..15, so no two sub-blocks
    share a scale or a min.
    """
    scales = bytes(((j + 1) % 16) | ((j + 2) % 16) << 4 for j in range(16))
    qs = bytes(((l + 3 * s) % 4) for s in range(2) for l in range(32))
    return scales + qs + struct.pack("<e", 0.5) + struct.pack("<e", 0.25)


def pack_k3_scales(values):
    """Pack sixteen 6-bit scales into Q3_K's twelve bytes.

    Scale i below eight takes the low nibble of byte i, from eight on the
    high nibble of byte i - 8. Its two high bits are one field of byte
    eight plus i % 4, at bit 2 * (i // 4).
    """
    out = bytearray(12)
    for i, v in enumerate(values):
        low, high = v & 0x0F, (v >> 4) & 0x03
        if i < 8:
            out[i] |= low
        else:
            out[i - 8] |= low << 4
        out[8 + (i % 4)] |= high << (2 * (i // 4))
    return bytes(out)


def q3_k_block():
    """One 110-byte Q3_K super-block of 256 weights.

    Thirty-two high-bit bytes, 64 bytes of 2-bit weights, twelve packed
    scale bytes and f16 d. The mask alternates so both branches of the
    three-bit weight are covered, and the sixteen scales span 3..45 so the
    six-bit fields differ after the minus thirty-two.
    """
    hmask = bytes((l * 7 + 1) & 0xFF for l in range(32))
    qs = bytes(((l + 2 * s) % 4) for s in range(2) for l in range(32))
    scales = pack_k3_scales([3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 5, 11, 17, 23, 29, 35])
    return hmask + qs + scales + struct.pack("<e", 1.0)


def write_container(path):
    qs = bytearray(16)
    for j in range(16):
        qs[j] = j | (((j + 8) % 16) << 4)
    block = struct.pack("<H", 0x3800) + bytes(qs)
    # Q5_0: scale 0.5, the low nibble of byte j carries weight j and the high
    # nibble weight j+16; qh holds the fifth bit, all ones for j < 16.
    q5_qs = bytes((j & 0x0F) | (((15 - j) & 0x0F) << 4) for j in range(16))
    block_q5 = struct.pack("<H", 0x3800) + struct.pack("<I", 0x0000FFFF) + q5_qs
    # Q8_0: scale 0.5 and thirty-two signed bytes spanning both signs.
    q8_qs = bytes((j * 5 - 16) & 0xFF for j in range(32))
    block_q8 = struct.pack("<H", 0x3800) + q8_qs
    f32 = b"".join(struct.pack("<f", i * 0.25) for i in range(32))

    tensors = [
        ("blk.0.test_q4_0", [32], 2, 0, block),
        ("blk.0.test_q5_0", [32], 6, 32, block_q5),
        ("blk.0.test_q8_0", [32], 8, 64, block_q8),
        ("blk.0.test_f32", [32], 0, 128, f32),
        ("blk.0.test_q4_k", [256], 12, 256, q4_k_block()),
        ("blk.0.test_q6_k", [256], 14, 416, q6_k_block()),
        ("blk.0.test_q5_k", [256], 13, 640, q5_k_block()),
        ("blk.0.test_q2_k", [256], 10, 816, q2_k_block()),
        ("blk.0.test_q3_k", [256], 11, 900, q3_k_block()),
    ]

    out = bytearray()
    out += MAGIC
    out += struct.pack("<I", 3)
    out += struct.pack("<Q", len(tensors))
    out += struct.pack("<Q", 2)
    for key, value in (("general.architecture", "llama"), ("general.name", "lowprec-fixture")):
        out += struct.pack("<Q", len(key)) + key.encode()
        out += struct.pack("<I", T_STRING)
        out += struct.pack("<Q", len(value)) + value.encode()
    for name, dims, kind, offset, _ in tensors:
        out += struct.pack("<Q", len(name)) + name.encode()
        out += struct.pack("<I", len(dims))
        for d in dims:
            out += struct.pack("<Q", d)
        out += struct.pack("<I", kind)
        out += struct.pack("<Q", offset)
    while len(out) % ALIGNMENT != 0:
        out += b"\x00"
    payload_base = len(out)
    for name, dims, kind, offset, payload in tensors:
        target = payload_base + offset
        while len(out) < target:
            out += b"\x00"
        out += payload
    with open(path, "wb") as f:
        f.write(out)
    return bytes(out)


def main():
    real = sys.argv[1]
    with open(real, "rb") as f:
        whole = f.read()
    print(f"// real file sha256 {hashlib.sha256(whole).hexdigest()} size {len(whole)}")
    version, alignment, kv, tensors, data_start = parse_container(whole)
    print(f"// real container: v{version} alignment {alignment} "
          f"kv {len(kv)} tensors {len(tensors)} data_start {data_start}")
    kinds = {}
    for t in tensors:
        kinds[t["kind"]] = kinds.get(t["kind"], 0) + 1
    print(f"// real tensor kinds {kinds}, first names {[t['name'] for t in tensors[:3]]}")

    head = whole[:24]
    print(f"// real header magic {head[:4].decode()} version "
          f"{int.from_bytes(head[4:8], 'little')} tensors "
          f"{int.from_bytes(head[8:16], 'little')} kv {int.from_bytes(head[16:24], 'little')}")

    body = write_container("fixture.gguf")
    readback = parse_container(body)
    print(f"// fixture: {len(body)} bytes, v{readback[0]} alignment {readback[1]} "
          f"data_start {readback[4]} tensors {[t['name'] for t in readback[3]]}")
    print(f"\nFIXTURE_HEX = {body.hex()}")
    print(f"REAL_HEADER_HEX = {head.hex()}")


if __name__ == "__main__":
    main()
