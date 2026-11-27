/// A representative set for binary16: zero, the first subnormals, the
/// smallest normal, every exponent decade up to the largest finite value,
/// the rounding boundary at 1.0, and the negative halves of each.
let f16_probe : Array[Int] = [
  0x0000, 0x0001, 0x0002, 0x00FF, 0x03FF, 0x0400, 0x0401, 0x0800,
  0x1400, 0x2000, 0x3000, 0x3555, 0x3800, 0x3C00, 0x3C01, 0x3E00,
  0x4000, 0x4200, 0x4400, 0x4800, 0x4C00, 0x5000, 0x5800, 0x6000,
  0x6800, 0x7BFF, 0x7C00, 0x7C01, 0x8000, 0x8001, 0xBC00, 0xC000,
  0xC400, 0xD000, 0xE000, 0xF800, 0xFBFF, 0xFC00,
]

fn same_or_both_nan(a : Int, b : Int) -> Bool {
  if a == b {
    true
  } else {
    F16::from_bits(a).is_nan() && F16::from_bits(b).is_nan()
  }
}

/// The double path is not the implementation under test; it is an
/// independent route to the same value. Where both agree, the integer path
/// is confirmed; where they disagree, at least one is wrong and the
/// disagreement is a finding.
test "f16 add agrees with the double path" {
  let mut checked = 0
  let mut mismatch = 0
  for i = 0; i < f16_probe.length(); i = i + 1 {
    for j = 0; j < f16_probe.length(); j = j + 1 {
      let a = f16_probe[i]
      let b = f16_probe[j]
      let exact = F16::from_bits(a).to_float() + F16::from_bits(b).to_float()
      let expected = F16::from_float(exact).bits()
      let actual = f16_add_bits(a, b)
      checked = checked + 1
      if !same_or_both_nan(actual, expected) {
        mismatch = mismatch + 1
      }
    }
  }
  assert_eq(checked, 1444)
  assert_eq(mismatch, 0)
}

test "f16 multiply agrees with the double path" {
  let mut checked = 0
  let mut mismatch = 0
  for i = 0; i < f16_probe.length(); i = i + 1 {
    for j = 0; j < f16_probe.length(); j = j + 1 {
      let a = f16_probe[i]
      let b = f16_probe[j]
      let exact = F16::from_bits(a).to_float() * F16::from_bits(b).to_float()
      let expected = F16::from_float(exact).bits()
      let actual = f16_mul_bits(a, b)
      checked = checked + 1
      if !same_or_both_nan(actual, expected) {
        mismatch = mismatch + 1
      }
    }
  }
  assert_eq(checked, 1444)
  assert_eq(mismatch, 0)
}

test "f16 divide agrees with the double path" {
  let mut checked = 0
  let mut mismatch = 0
  for i = 0; i < f16_probe.length(); i = i + 1 {
    for j = 0; j < f16_probe.length(); j = j + 1 {
      let a = f16_probe[i]
      let b = f16_probe[j]
      let exact = F16::from_bits(a).to_float() / F16::from_bits(b).to_float()
      let expected = F16::from_float(exact).bits()
      let actual = f16_div_bits(a, b)
      checked = checked + 1
      if !same_or_both_nan(actual, expected) {
        mismatch = mismatch + 1
      }
    }
  }
  assert_eq(checked, 1444)
  assert_eq(mismatch, 0)
}

test "f16 subtract agrees with the double path" {
  let mut checked = 0
  for i = 0; i < f16_probe.length(); i = i + 1 {
    for j = 0; j < f16_probe.length(); j = j + 1 {
      let a = f16_probe[i]
      let b = f16_probe[j]
      let exact = F16::from_bits(a).to_float() - F16::from_bits(b).to_float()
      let expected = F16::from_float(exact).bits()
      let actual = f16_sub_bits(a, b)
      checked = checked + 1
      assert_eq(same_or_both_nan(actual, expected), true)
    }
  }
  assert_eq(checked, 1444)
}

test "f16 special values follow ieee 754" {
  let one = F16::from_bits(0x3C00)
  let inf = F16::from_bits(0x7C00)
  let qnan = F16::from_bits(0x7E00)
  assert_eq(one.div(F16::from_bits(0x0000)).bits(), 0x7C00)
  assert_eq(F16::from_bits(0x0000).div(F16::from_bits(0x0000)).is_nan(), true)
  assert_eq(inf.add(F16::from_bits(0xFC00)).is_nan(), true)
  assert_eq(inf.mul(F16::from_bits(0x0000)).is_nan(), true)
  assert_eq(qnan.add(one).is_nan(), true)
  // 65504 + 65504 overflows to infinity rather than wrapping.
  assert_eq(F16::from_bits(0x7BFF).add(F16::from_bits(0x7BFF)).bits(), 0x7C00)
  // (-0) + (-0) keeps the sign; every other zero sum is +0.
  assert_eq(F16::from_bits(0x8000).add(F16::from_bits(0x8000)).bits(), 0x8000)
  assert_eq(one.add(F16::from_bits(0xBC00)).bits(), 0x0000)
}
