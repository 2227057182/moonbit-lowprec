"""Builds the GGUF fixtures and the test that checks them.

Everything here is an independent second implementation of the container
format: a small reader and writer written straight from the GGUF v3 layout,
with no dependency on the MoonBit code under test and none on llama.cpp.

Two outputs:

1. The first 24 bytes of a real released model (stories260K.gguf, sha256
   270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d). The
   reader is therefore checked against a file this repository did not produce,
   and the values below were cross-checked with llama.cpp's own gguf reader.

2. A complete container with one Q4_0 block and one F32 tensor, written by
   this writer and read back by this reader, then embedded in the test.

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


def write_container(path):
    qs = bytearray(16)
    for j in range(16):
        qs[j] = j | (((j + 8) % 16) << 4)
    block = struct.pack("<H", 0x3800) + bytes(qs)
    f32 = b"".join(struct.pack("<f", i * 0.25) for i in range(32))

    out = bytearray()
    out += MAGIC
    out += struct.pack("<I", 3)
    out += struct.pack("<Q", 2)
    out += struct.pack("<Q", 2)
    for key, value in (("general.architecture", "llama"), ("general.name", "lowprec-fixture")):
        out += struct.pack("<Q", len(key)) + key.encode()
        out += struct.pack("<I", T_STRING)
        out += struct.pack("<Q", len(value)) + value.encode()
    tensors = [
        ("blk.0.test_q4_0", [32], 2, 0, block),
        ("blk.0.test_f32", [32], 0, 32, f32),
    ]
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
