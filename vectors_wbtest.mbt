test "published vectors reproduce" {
  let (checked, failed) = verify_vectors()
  assert_eq(checked, 60)
  assert_eq(failed, 0)
}

test "vector table covers the extremes of every type" {
  // binary16: the smallest subnormal and the largest finite value are both
  // in the table, so a downstream reader can pin both ends.
  assert_eq(f16_vectors[1].0, 0x0001)
  assert_eq(f16_vectors[1].1, 0.000000059604644775390625)
  assert_eq(f16_vectors[19].0, 0x7BFF)
  assert_eq(f16_vectors[19].1, 65504.0)
  // bfloat16: same exponent range as binary32.
  assert_eq(bf16_vectors[11].0, 0x7F7F)
  // quantization: both int8 endpoints.
  assert_eq(quant_vectors[0].0, -128)
  assert_eq(quant_vectors[10].0, 127)
}
