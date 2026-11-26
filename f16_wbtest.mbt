/// Exhaustive check: every binary16 pattern except NaN must survive the
/// widening to binary32 and narrowing back unchanged. This covers all
/// 65536 patterns, so no value is left untested.
test "exhaustive roundtrip over all 65536 patterns" {
  let mut checked = 0
  for b = 0; b <= 0xFFFF; b = b + 1 {
    let v = F16::from_bits(b)
    if v.is_nan() {
      continue
    }
    let back = F16::from_float(v.to_float())
    assert_eq(back.bits(), b)
    checked = checked + 1
  }
  // 65536 patterns minus the 2046 NaN payloads.
  assert_eq(checked, 63490)
}

/// Widening must be monotone over the positive finite range: a larger bit
/// pattern means a larger value. Catches any mis-normalised subnormal.
test "widening is monotone over positive finite values" {
  let mut prev = F16::from_bits(0).to_float()
  for b = 0; b <= 0x7BFF; b = b + 1 {
    let x = F16::from_bits(b).to_float()
    assert_true(x >= prev)
    prev = x
  }
}

/// Spot values with hand-checked bit patterns.
test "known values from the binary16 table" {
  assert_eq(F16::from_bits(0x3C00).to_float(), 1.0)
  assert_eq(F16::from_bits(0x4000).to_float(), 2.0)
  assert_eq(F16::from_bits(0xC000).to_float(), -2.0)
  assert_eq(F16::from_bits(0x7BFF).to_float(), 65504.0)
  assert_eq(F16::from_bits(0x0001).to_float(), 1.0 / 16777216.0)
  assert_eq(F16::from_bits(0x0400).to_float(), 1.0 / 16384.0)
  assert_eq(F16::from_bits(0x0000).to_float(), 0.0)
  assert_eq(F16::from_bits(0x8000).to_float(), 0.0)
  assert_eq(F16::from_bits(0x1400).to_float(), 1.0 / 1024.0)
}

/// Classification must agree with the raw fields on every pattern.
test "classification over all patterns" {
  for b = 0; b <= 0xFFFF; b = b + 1 {
    let v = F16::from_bits(b)
    let e = (b >> 10) & 0x1F
    let m = b & 0x3FF
    assert_eq(v.is_nan(), e == 31 && m != 0)
    assert_eq(v.is_infinite(), e == 31 && m == 0)
    assert_eq(v.is_zero(), (b & 0x7FFF) == 0)
    assert_eq(v.is_subnormal(), e == 0 && m != 0)
    assert_eq(v.is_normal(), e != 0 && e != 31)
    assert_eq(v.is_finite(), e != 31)
  }
}

/// Narrowing ties must go to the even significand, not always up.
test "round to nearest ties to even when narrowing" {
  // 1 + 2^-11 sits exactly halfway between 1 and 1 + 2^-10.
  assert_eq(F16::from_float(1.0 + 1.0 / 2048.0).bits(), 0x3C00)
  // 1 + 3*2^-11 sits exactly halfway between 1 + 2^-10 and 1 + 2*2^-10,
  // so the even significand wins.
  assert_eq(F16::from_float(1.0 + 3.0 / 2048.0).bits(), 0x3C02)
  // A value just above the halfway point must round up.
  assert_eq(F16::from_float(1.0 + 1.0 / 2048.0 + 1.0 / 8388608.0).bits(), 0x3C01)
}

/// Overflow and underflow behave as the standard requires.
test "overflow to infinity and underflow to zero" {
  assert_eq(F16::from_float(65504.0).bits(), 0x7BFF)
  assert_eq(F16::from_float(65520.0).bits(), 0x7C00)
  assert_eq(F16::from_float(-70000.0).bits(), 0xFC00)
  assert_eq(F16::from_float(1.0 / 16777216.0).bits(), 0x0001)
  assert_eq(F16::from_float(1.0 / 33554432.0).bits(), 0x0000)
  assert_eq(F16::from_float(0.0).bits(), 0x0000)
  assert_eq(F16::from_float(-0.0).bits(), 0x8000)
}

/// Special values round-trip through the public constructors.
test "special value constructors" {
  assert_true(min_subnormal().is_subnormal())
  assert_true(min_normal().is_normal())
  assert_true(max_value().is_finite())
  assert_true(infinity().is_infinite())
  assert_true(nan().is_nan())
  assert_eq(precision(), 11)
  assert_eq(epsilon().to_float(), 1.0 / 1024.0)
}

/// Masking keeps only the low sixteen bits.
test "from_bits masks high bits" {
  assert_eq(F16::from_bits(0x1FFFF).bits(), 0xFFFF)
  assert_eq(F16::from_bits(-1).bits(), 0xFFFF)
}
