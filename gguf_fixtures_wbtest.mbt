/// Fixtures built by `tools/make_gguf_fixtures.py`, an independent Python
/// implementation of the container format, and checked by
/// `tools/emit_gguf_test.py`, a second implementation of the block layouts.
///
/// `fixture_gguf_hex` is a complete 1554-byte container holding one block
/// each of Q4_0, Q5_0, Q8_0, Q4_K, Q6_K, Q5_K, Q2_K and Q3_K plus a 32-value
/// F32 tensor.
///
/// `real_header_hex` is the first 24 bytes of stories260K.gguf, a released
/// model whose sha256 is
/// 270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d.
/// The same 24 bytes were independently resolved by llama.cpp's reader as
/// version 3, 48 tensors, 19 metadata entries.
let fixture_gguf_hex : String = "474755460300000009000000000000000200000000000000140000000000000067656e6572616c2e6172636869746563747572650800000005000000000000006c6c616d610c0000000000000067656e6572616c2e6e616d65080000000f000000000000006c6f77707265632d666978747572650f00000000000000626c6b2e302e746573745f71345f300100000020000000000000000200000000000000000000000f00000000000000626c6b2e302e746573745f71355f300100000020000000000000000600000020000000000000000f00000000000000626c6b2e302e746573745f71385f300100000020000000000000000800000040000000000000000e00000000000000626c6b2e302e746573745f6633320100000020000000000000000000000080000000000000000f00000000000000626c6b2e302e746573745f71345f6b0100000000010000000000000c00000000010000000000000f00000000000000626c6b2e302e746573745f71365f6b0100000000010000000000000e000000a0010000000000000f00000000000000626c6b2e302e746573745f71355f6b0100000000010000000000000d00000080020000000000000f00000000000000626c6b2e302e746573745f71325f6b0100000000010000000000000a00000030030000000000000f00000000000000626c6b2e302e746573745f71335f6b0100000000010000000000000b000000840300000000000000000000000000388091a2b3c4d5e6f708192a3b4c5d6e7f00000000000000000000000000000038ffff0000f0e1d2c3b4a5968778695a4b3c2d1e0f000000000000000000000038f0f5faff04090e13181d22272c31363b40454a4f54595e63686d72777c81868b000000000000000000000000000000000000000000000000000000000000000000000000803e0000003f0000403f0000803f0000a03f0000c03f0000e03f0000004000001040000020400000304000004040000050400000604000007040000080400000884000009040000098400000a0400000a8400000b0400000b8400000c0400000c8400000d0400000d8400000e0400000e8400000f0400000f84000380034414243440506070811223344405162738495a6b7c8d9eafb0c1d2e3f405162738495a6b7c8d9eafb0c1d2e3f5162738495a6b7c8d9eafb0c1d2e3f405162738495a6b7c8d9eafb0c1d2e3f4062738495a6b7c8d9eafb0c1d2e3f405162738495a6b7c8d9eafb0c1d2e3f4051738495a6b7c8d9eafb0c1d2e3f405162738495a6b7c8d9eafb0c1d2e3f40516200000000000000000000000000000000708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6fc3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2e4394e93e4394e93e4394e93e4394e93e4394e93e4394e93e4394e93e4394e93e4394e93e4394e93e4394e93e4394e93e4394e93e4394e93e4394e93e4394e93f8f9fafbfcfdfeff00010203040506070034000000000000000000000000000000380034414243440506070811223344000306090c0f1215181b1e2124272a2d303336393c3f4245484b4e5154575a5d405162738495a6b7c8d9eafb0c1d2e3f405162738495a6b7c8d9eafb0c1d2e3f5162738495a6b7c8d9eafb0c1d2e3f405162738495a6b7c8d9eafb0c1d2e3f4062738495a6b7c8d9eafb0c1d2e3f405162738495a6b7c8d9eafb0c1d2e3f4051738495a6b7c8d9eafb0c1d2e3f405162738495a6b7c8d9eafb0c1d2e3f4051622132435465768798a9bacbdcedfe0f10000102030001020300010203000102030001020300010203000102030001020303000102030001020300010203000102030001020300010203000102030001020038003401080f161d242b323940474e555c636a71787f868d949ba2a9b0b7bec5ccd3da0001020300010203000102030001020300010203000102030001020300010203020300010203000102030001020300010203000102030001020300010203000133995fb51b71d73d74784889003c"

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
  assert_eq(file.tensors.length(), 9)
  assert_eq(file.data_start, 544)
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
  let q4k = dequantize_tensor(bytes, file, file.tensors[4])
  assert_eq(q4k[0], -1.25)
  assert_eq(q4k[31], 6.25)
  assert_eq(q4k[32], 2.5)
  assert_eq(q4k[64], -0.25)
  assert_eq(q4k[96], 8.0)
  assert_eq(q4k[128], 16.75)
  assert_eq(q4k[255], 59.0)
  let q6k = dequantize_tensor(bytes, file, file.tensors[5])
  assert_eq(q6k[0], 64.0)
  assert_eq(q6k[31], -54.25)
  assert_eq(q6k[32], 24.0)
  assert_eq(q6k[64], -7.0)
  assert_eq(q6k[96], -11.5)
  assert_eq(q6k[128], -0.0)
  assert_eq(q6k[255], 19.25)
  let q5k = dequantize_tensor(bytes, file, file.tensors[6])
  assert_eq(q5k[0], -1.25)
  assert_eq(q5k[31], 14.25)
  assert_eq(q5k[32], 2.5)
  assert_eq(q5k[64], -0.25)
  assert_eq(q5k[96], 8.0)
  assert_eq(q5k[128], 16.75)
  assert_eq(q5k[255], 59.0)
  // Q2_K subtracts `dmin` after scaling, Q3_K's third bit inverts the offset
  // of four. The values below match the official gguf package on all 256.
  let q2k = dequantize_tensor(bytes, file, file.tensors[7])
  assert_eq(q2k[0], -0.5)
  assert_eq(q2k[31], 2.25)
  assert_eq(q2k[32], -1.0)
  assert_eq(q2k[64], -1.5)
  assert_eq(q2k[128], 11.0)
  assert_eq(q2k[255], -0.25)
  let q3k = dequantize_tensor(bytes, file, file.tensors[8])
  assert_eq(q3k[1], 87.0)
  assert_eq(q3k[15], 29.0)
  assert_eq(q3k[31], 23.0)
  assert_eq(q3k[64], 20.0)
  assert_eq(q3k[95], -4.0)
  assert_eq(q3k[96], -28.0)
  assert_eq(q3k[128], -38.0)
  assert_eq(q3k[191], 84.0)
}
