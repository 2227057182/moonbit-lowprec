/// The exact reference table, reproduced bit for bit.
///
/// The table is built from rational arithmetic, not from an emulator, so
/// agreement here is agreement with the standard's definition rather than
/// agreement with another implementation's habits. The same binary16 pairs
/// were also run through numpy's float16 and agreed on all 684 results, which
/// is what makes the table trustworthy as a baseline.
test "binary16 arithmetic reproduces the exact reference table" {
  let mut checked = 0
  for i = 0; i < f16_arithmetic_vectors.length(); i = i + 1 {
    let v = f16_arithmetic_vectors[i]
    assert_eq(f16_add_bits(v.a, v.b), v.sum)
    assert_eq(f16_sub_bits(v.a, v.b), v.difference)
    assert_eq(f16_mul_bits(v.a, v.b), v.product)
    assert_eq(f16_div_bits(v.a, v.b), v.quotient)
    checked = checked + 4
  }
  assert_eq(checked, 684)
}

test "bfloat16 arithmetic reproduces the exact reference table" {
  let mut checked = 0
  for i = 0; i < bf16_arithmetic_vectors.length(); i = i + 1 {
    let v = bf16_arithmetic_vectors[i]
    assert_eq(bf16_add_bits(v.a, v.b), v.sum)
    assert_eq(bf16_sub_bits(v.a, v.b), v.difference)
    assert_eq(bf16_mul_bits(v.a, v.b), v.product)
    assert_eq(bf16_div_bits(v.a, v.b), v.quotient)
    checked = checked + 4
  }
  assert_eq(checked, 684)
}

/// The table must actually reach the boundaries it claims, otherwise
/// agreement means little.
test "the reference table covers the boundaries" {
  let mut subnormal_results = 0
  let mut infinite_results = 0
  for i = 0; i < f16_arithmetic_vectors.length(); i = i + 1 {
    let v = f16_arithmetic_vectors[i]
    for bits in [v.sum, v.difference, v.product, v.quotient] {
      let e = (bits >> 10) & 31
      let m = bits & 1023
      if e == 0 && m != 0 {
        subnormal_results = subnormal_results + 1
      }
      if e == 31 {
        infinite_results = infinite_results + 1
      }
    }
  }
  assert_eq(subnormal_results > 40, true)
  assert_eq(infinite_results > 4, true)
}
