/// The same probe idea for bfloat16: eight exponent bits mean the subnormal
/// range is far below anything a float32 exponent can reach, so the double
/// path is still an independent check there.
let bf16_probe : Array[Int] = [
  0x0000, 0x0001, 0x0002, 0x0010, 0x007F, 0x0080, 0x0081, 0x0100,
  0x3F80, 0x3F81, 0x3FA0, 0x4000, 0x4040, 0x4080, 0x4100, 0x4180,
  0x4200, 0x4400, 0x4600, 0x4800, 0x5000, 0x6000, 0x7000, 0x7F7F,
  0x7F80, 0x7FC0, 0x8000, 0x8001, 0xBF80, 0xC000, 0xC040, 0xC200,
  0xD000, 0xE000, 0xF000, 0xFF7F, 0xFF80, 0xFFC0,
]

fn bf16_same_or_both_nan(a : Int, b : Int) -> Bool {
  if a == b {
    true
  } else {
    BF16::from_bits(a).is_nan() && BF16::from_bits(b).is_nan()
  }
}

test "bf16 add agrees with the double path" {
  let mut checked = 0
  let mut mismatch = 0
  for i = 0; i < bf16_probe.length(); i = i + 1 {
    for j = 0; j < bf16_probe.length(); j = j + 1 {
      let a = bf16_probe[i]
      let b = bf16_probe[j]
      let exact = BF16::from_bits(a).to_float() + BF16::from_bits(b).to_float()
      let expected = BF16::from_float(exact).bits()
      let actual = bf16_add_bits(a, b)
      checked = checked + 1
      if !bf16_same_or_both_nan(actual, expected) {
        mismatch = mismatch + 1
      }
    }
  }
  assert_eq(checked, 1444)
  assert_eq(mismatch, 0)
}

test "bf16 multiply agrees with the double path" {
  let mut checked = 0
  let mut mismatch = 0
  for i = 0; i < bf16_probe.length(); i = i + 1 {
    for j = 0; j < bf16_probe.length(); j = j + 1 {
      let a = bf16_probe[i]
      let b = bf16_probe[j]
      let exact = BF16::from_bits(a).to_float() * BF16::from_bits(b).to_float()
      let expected = BF16::from_float(exact).bits()
      let actual = bf16_mul_bits(a, b)
      checked = checked + 1
      if !bf16_same_or_both_nan(actual, expected) {
        mismatch = mismatch + 1
      }
    }
  }
  assert_eq(checked, 1444)
  assert_eq(mismatch, 0)
}

test "bf16 divide agrees with the double path" {
  let mut checked = 0
  let mut mismatch = 0
  for i = 0; i < bf16_probe.length(); i = i + 1 {
    for j = 0; j < bf16_probe.length(); j = j + 1 {
      let a = bf16_probe[i]
      let b = bf16_probe[j]
      let exact = BF16::from_bits(a).to_float() / BF16::from_bits(b).to_float()
      let expected = BF16::from_float(exact).bits()
      let actual = bf16_div_bits(a, b)
      checked = checked + 1
      if !bf16_same_or_both_nan(actual, expected) {
        mismatch = mismatch + 1
      }
    }
  }
  assert_eq(checked, 1444)
  assert_eq(mismatch, 0)
}

test "bf16 subtract agrees with the double path" {
  let mut checked = 0
  for i = 0; i < bf16_probe.length(); i = i + 1 {
    for j = 0; j < bf16_probe.length(); j = j + 1 {
      let a = bf16_probe[i]
      let b = bf16_probe[j]
      let exact = BF16::from_bits(a).to_float() - BF16::from_bits(b).to_float()
      let expected = BF16::from_float(exact).bits()
      let actual = bf16_sub_bits(a, b)
      checked = checked + 1
      assert_eq(bf16_same_or_both_nan(actual, expected), true)
    }
  }
  assert_eq(checked, 1444)
}

test "bf16 special values follow ieee 754" {
  let one = BF16::from_bits(0x3F80)
  let inf = BF16::from_bits(0x7F80)
  assert_eq(inf.add(BF16::from_bits(0xFF80)).is_nan(), true)
  assert_eq(inf.mul(BF16::from_bits(0x0000)).is_nan(), true)
  assert_eq(one.div(BF16::from_bits(0x0000)).bits(), 0x7F80)
  assert_eq(BF16::from_bits(0x8000).add(BF16::from_bits(0x8000)).bits(), 0x8000)
}
