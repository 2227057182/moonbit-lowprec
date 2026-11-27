/// Fixtures built by `tools/make_gguf_fixtures.py`, an independent Python
/// implementation of the container format, and checked by
/// `tools/emit_gguf_test.py`, a second implementation of the block layouts.
///
/// `fixture_gguf_hex` is a complete 576-byte container holding one block
/// each of Q4_0, Q5_0 and Q8_0 plus a 32-value F32 tensor.
///
/// `real_header_hex` is the first 24 bytes of stories260K.gguf, a released
/// model whose sha256 is
/// 270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d.
/// The same 24 bytes were independently resolved by llama.cpp's reader as
/// version 3, 48 tensors, 19 metadata entries.
let fixture_gguf_hex : String = "474755460300000004000000000000000200000000000000140000000000000067656e6572616c2e6172636869746563747572650800000005000000000000006c6c616d610c0000000000000067656e6572616c2e6e616d65080000000f000000000000006c6f77707265632d666978747572650f00000000000000626c6b2e302e746573745f71345f300100000020000000000000000200000000000000000000000f00000000000000626c6b2e302e746573745f71355f300100000020000000000000000600000020000000000000000f00000000000000626c6b2e302e746573745f71385f300100000020000000000000000800000040000000000000000e00000000000000626c6b2e302e746573745f663332010000002000000000000000000000008000000000000000000000000000000000000000000000000000388091a2b3c4d5e6f708192a3b4c5d6e7f00000000000000000000000000000038ffff0000f0e1d2c3b4a5968778695a4b3c2d1e0f000000000000000000000038f0f5faff04090e13181d22272c31363b40454a4f54595e63686d72777c81868b000000000000000000000000000000000000000000000000000000000000000000000000803e0000003f0000403f0000803f0000a03f0000c03f0000e03f0000004000001040000020400000304000004040000050400000604000007040000080400000884000009040000098400000a0400000a8400000b0400000b8400000c0400000c8400000d0400000d8400000e0400000e8400000f0400000f840"

let real_header_hex : String = "474755460300000030000000000000001300000000000000"

test "the fixed header of a released model is read correctly" {
  let bytes = bytes_from_hex(real_header_hex)
  let header = parse_gguf_header(bytes).unwrap()
  assert_eq(header.version, 3)
  assert_eq(header.tensor_count, 48)
  assert_eq(header.kv_count, 19)
}

test "a short buffer is enough to identify a file" {
  // The same 24 bytes with the magic broken must be rejected, so the check
  // is on content and not only on length.
  let bytes = bytes_from_hex(real_header_hex)
  bytes[1] = Int::to_byte(88)
  assert_eq(parse_gguf_header(bytes) is None, true)
}

test "a fixture container parses and dequantizes" {
  let bytes = bytes_from_hex(fixture_gguf_hex)
  let file = parse_gguf(bytes).unwrap()
  assert_eq(file.version, 3)
  assert_eq(file.tensors.length(), 4)
  assert_eq(file.data_start, 320)
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
  assert_eq(q4[0], -4.0)
  assert_eq(q4[15], 3.5)
  assert_eq(q4[16], 0.0)
  assert_eq(q4[31], -0.5)
  let q5 = dequantize_tensor(bytes, file, file.tensors[1])
  assert_eq(q5[0], 0.0)
  assert_eq(q5[15], 7.5)
  assert_eq(q5[16], -0.5)
  assert_eq(q5[31], -8.0)
  let q8 = dequantize_tensor(bytes, file, file.tensors[2])
  assert_eq(q8[0], -8.0)
  assert_eq(q8[7], 9.5)
  assert_eq(q8[31], -58.5)
  let w = dequantize_tensor(bytes, file, file.tensors[3])
  assert_eq(w[0], 0.0)
  assert_eq(w[7], 1.75)
  assert_eq(w[31], 7.75)
}
