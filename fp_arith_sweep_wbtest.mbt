/// Every one of the 65536 binary16 patterns, against a dozen probes that
/// cover zero, the subnormal floor, the smallest normal, 1.0 and its
/// neighbour, a round number, the largest finite value, and their negatives.
/// This is the sweep that a hand-picked example set cannot give: any pattern
/// that decodes or re-encodes wrongly shows up here.
let sweep_probes : Array[Int] = [
  0x0000, 0x0001, 0x03FF, 0x0400, 0x3C00, 0x3C01,
  0x3E00, 0x4000, 0x7BFF, 0x8000, 0xC000, 0xFBFF,
]

test "f16 add over all patterns against the probes" {
  let mut checked = 0
  let mut mismatch = 0
  for a = 0; a < 65536; a = a + 1 {
    for k = 0; k < sweep_probes.length(); k = k + 1 {
      let b = sweep_probes[k]
      let exact = F16::from_bits(a).to_float() + F16::from_bits(b).to_float()
      let expected = F16::from_float(exact).bits()
      let actual = f16_add_bits(a, b)
      checked = checked + 1
      if !same_or_both_nan(actual, expected) {
        mismatch = mismatch + 1
      }
    }
  }
  assert_eq(checked, 786432)
  assert_eq(mismatch, 0)
}

test "f16 multiply over all patterns against the probes" {
  let mut checked = 0
  let mut mismatch = 0
  for a = 0; a < 65536; a = a + 1 {
    for k = 0; k < sweep_probes.length(); k = k + 1 {
      let b = sweep_probes[k]
      let exact = F16::from_bits(a).to_float() * F16::from_bits(b).to_float()
      let expected = F16::from_float(exact).bits()
      let actual = f16_mul_bits(a, b)
      checked = checked + 1
      if !same_or_both_nan(actual, expected) {
        mismatch = mismatch + 1
      }
    }
  }
  assert_eq(checked, 786432)
  assert_eq(mismatch, 0)
}

test "f16 divide and subtract over all patterns against the probes" {
  let mut checked = 0
  let mut mismatch = 0
  for a = 0; a < 65536; a = a + 1 {
    for k = 0; k < sweep_probes.length(); k = k + 1 {
      let b = sweep_probes[k]
      let va = F16::from_bits(a).to_float()
      let vb = F16::from_bits(b).to_float()
      let want_quotient = F16::from_float(va / vb).bits()
      let want_difference = F16::from_float(va - vb).bits()
      let got_quotient = f16_div_bits(a, b)
      let got_difference = f16_sub_bits(a, b)
      checked = checked + 2
      if !same_or_both_nan(got_quotient, want_quotient) {
        mismatch = mismatch + 1
      }
      if !same_or_both_nan(got_difference, want_difference) {
        mismatch = mismatch + 1
      }
    }
  }
  assert_eq(checked, 1572864)
  assert_eq(mismatch, 0)
}

test "bf16 add and multiply over all patterns against the probes" {
  let mut checked = 0
  let mut mismatch = 0
  for a = 0; a < 65536; a = a + 1 {
    for k = 0; k < sweep_probes.length(); k = k + 1 {
      let b = sweep_probes[k]
      let one_a = BF16::from_bits(a).to_float()
      let one_b = BF16::from_bits(b).to_float()
      let want_sum = BF16::from_float(one_a + one_b).bits()
      let want_product = BF16::from_float(one_a * one_b).bits()
      let got_sum = bf16_add_bits(a, b)
      let got_product = bf16_mul_bits(a, b)
      checked = checked + 2
      if !bf16_same_or_both_nan(got_sum, want_sum) {
        mismatch = mismatch + 1
      }
      if !bf16_same_or_both_nan(got_product, want_product) {
        mismatch = mismatch + 1
      }
    }
  }
  assert_eq(checked, 1572864)
  assert_eq(mismatch, 0)
}

test "bf16 divide and subtract over all patterns against the probes" {
  let mut checked = 0
  let mut mismatch = 0
  for a = 0; a < 65536; a = a + 1 {
    for k = 0; k < sweep_probes.length(); k = k + 1 {
      let b = sweep_probes[k]
      let one_a = BF16::from_bits(a).to_float()
      let one_b = BF16::from_bits(b).to_float()
      let want_quotient = BF16::from_float(one_a / one_b).bits()
      let want_difference = BF16::from_float(one_a - one_b).bits()
      let got_quotient = bf16_div_bits(a, b)
      let got_difference = bf16_sub_bits(a, b)
      checked = checked + 2
      if !bf16_same_or_both_nan(got_quotient, want_quotient) {
        mismatch = mismatch + 1
      }
      if !bf16_same_or_both_nan(got_difference, want_difference) {
        mismatch = mismatch + 1
      }
    }
  }
  assert_eq(checked, 1572864)
  assert_eq(mismatch, 0)
}
