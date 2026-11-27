test "the q2_k and q3_k payload sizes follow the super-block" {
  let bytes = bytes_from_hex(fixture_gguf_hex)
  let file = parse_gguf(bytes).unwrap()
  assert_eq(file.tensors[7].byte_size(), 84)
  assert_eq(file.tensors[8].byte_size(), 110)
}

test "the twelve q3_k scale bytes unpack to sixteen signed fields" {
  // The fixture packs 3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 5, 11, 17, 23,
  // 29, 35 and every reader subtracts thirty-two.
  let bytes = bytes_from_hex(fixture_gguf_hex)
  let file = parse_gguf(bytes).unwrap()
  let base = file.data_start + file.tensors[8].offset + 96
  let expected : Array[Int] = [
    -29, -23, -17, -11, -5, 1, 7, 13, 19, 25, -27, -21, -15, -9, -3, 3,
  ]
  for i = 0; i < 16; i = i + 1 {
    assert_eq(k3_scale(bytes, base, i), expected[i])
  }
}
