"""Emits the MoonBit test that checks the GGUF fixtures.

The hex strings are read from the files rather than retyped, so the test
cannot drift from the fixtures it claims to check.

    python3 tools/emit_gguf_test.py <path-to-real.gguf>
"""

import sys

FIXTURE = "fixture.gguf"
OUT = "gguf_fixtures_wbtest.mbt"

TEMPLATE = '''/// Fixtures built by `tools/make_gguf_fixtures.py`, an independent Python
/// implementation of the container format.
///
/// `fixture_gguf_hex` is a complete 402-byte container holding one Q4_0 block
/// and one F32 vector, written by that Python writer and read back by its
/// reader before being embedded here.
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
  assert_eq(file.tensors.length(), 2)
  assert_eq(file.data_start, 224)
  assert_eq(file.tensors[0].name, "blk.0.test_q4_0")
  assert_eq(file.tensors[0].kind, 2)
  assert_eq(file.tensors[0].dims[0], 32)
  assert_eq(file.tensors[1].name, "blk.0.test_f32")
  assert_eq(file.tensors[1].kind, 0)
  let q4 = dequantize_tensor(bytes, file, file.tensors[0])
  // scale is 0.5, the low nibble of byte j holds weight j and the high nibble
  // holds weight j+16, each biased by eight.
  assert_eq(q4[0], -4.0)
  assert_eq(q4[15], 3.5)
  assert_eq(q4[16], 0.0)
  assert_eq(q4[31], -0.5)
  let f32 = dequantize_tensor(bytes, file, file.tensors[1])
  assert_eq(f32[0], 0.0)
  assert_eq(f32[7], 1.75)
  assert_eq(f32[31], 7.75)
}}
'''


def main():
    real = sys.argv[1]
    with open(real, "rb") as f:
        head = f.read(24)
    with open(FIXTURE, "rb") as f:
        body = f.read()
    text = TEMPLATE.format(fixture=body.hex(), real=head.hex())
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {OUT}: {len(text)} bytes, fixture {len(body)} bytes")


if __name__ == "__main__":
    main()
