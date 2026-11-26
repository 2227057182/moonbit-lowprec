/// Exhaustive: every bfloat16 pattern except NaN must return to itself
/// after widening to binary32 and narrowing back with round-to-nearest.
test "exhaustive roundtrip over all 65536 bfloat16 patterns" {
  let mut checked = 0
  for b = 0; b <= 0xFFFF; b = b + 1 {
    let v = BF16::from_bits(b)
    if v.is_nan() {
      continue
    }
    let back = BF16::from_float(v.to_float())
    assert_eq(back.bits(), b)
    checked = checked + 1
  }
  // 65536 patterns minus the 254 NaN payloads (0x7F81..0x7FFF and 0xFF81..0xFFFF).
  assert_eq(checked, 65282)
}

/// Widening must be monotone across the positive finite range.
test "bfloat16 widening is monotone over positive finite values" {
  let mut prev = BF16::from_bits(0).to_float()
  for b = 0; b <= 0x7F7F; b = b + 1 {
    let x = BF16::from_bits(b).to_float()
    assert_true(x >= prev)
    prev = x
  }
}

/// Spot values with hand-checked patterns.
test "known bfloat16 values" {
  assert_eq(BF16::from_bits(0x3F80).to_float(), 1.0)
  assert_eq(BF16::from_bits(0x4000).to_float(), 2.0)
  assert_eq(BF16::from_bits(0xC000).to_float(), -2.0)
  assert_eq(BF16::from_bits(0x0000).to_float(), 0.0)
  // 2^-126 (smallest normal) and 2^-133 (smallest subnormal), checked by
  // the exact binary32 pattern instead of a rounded decimal literal.
  assert_eq(BF16::from_bits(0x0080).to_float(), Float::reinterpret_from_int(0x00800000))
  assert_eq(BF16::from_bits(0x0001).to_float(), Float::reinterpret_from_int(0x00010000))
  assert_eq(BF16::from_float(1.0).bits(), 0x3F80)
  assert_eq(BF16::from_float(2.0).bits(), 0x4000)
  assert_eq(BF16::from_float(-2.0).bits(), 0xC000)
}

/// Classification must agree with the raw fields on every pattern.
test "bfloat16 classification over all patterns" {
  for b = 0; b <= 0xFFFF; b = b + 1 {
    let v = BF16::from_bits(b)
    let e = (b >> 7) & 0xFF
    let m = b & 0x7F
    assert_eq(v.is_nan(), e == 0xFF && m != 0)
    assert_eq(v.is_infinite(), e == 0xFF && m == 0)
    assert_eq(v.is_zero(), (b & 0x7FFF) == 0)
    assert_eq(v.is_subnormal(), e == 0 && m != 0)
    assert_eq(v.is_normal(), e != 0 && e != 0xFF)
    assert_eq(v.is_finite(), e != 0xFF)
  }
}

/// Narrowing rounds ties to even, and the two modes differ where they should.
test "bfloat16 rounding modes" {
  // 1 + 2^-8 sits exactly halfway between 1 and 1 + 2^-7; the even one wins.
  assert_eq(BF16::from_float(1.0 + 1.0 / 256.0).bits(), 0x3F80)
  assert_eq(BF16::from_float_truncate(1.0 + 1.0 / 256.0).bits(), 0x3F80)
  // 1 + 3*2^-8 sits halfway between 1 + 2^-7 and 1 + 2*2^-7; even wins.
  assert_eq(BF16::from_float(1.0 + 3.0 / 256.0).bits(), 0x3F82)
  assert_eq(BF16::from_float_truncate(1.0 + 3.0 / 256.0).bits(), 0x3F81)
  // Just above a halfway point must round up. 2^-23 is one binary32 ulp at
  // 1.0, so the sum is representable; 2^-24 would already be absorbed and
  // leave the value sitting exactly on the tie.
  assert_eq(BF16::from_float(1.0 + 1.0 / 256.0 + 1.0 / 8388608.0).bits(), 0x3F81)
}

/// Special values keep their meaning through both narrowing modes.
test "bfloat16 special values" {
  assert_eq(BF16::from_float(Float::reinterpret_from_int(0x7F800000)).bits(), 0x7F80)
  assert_eq(BF16::from_float(Float::reinterpret_from_int(0xFF800000)).bits(), 0xFF80)
  assert_true(BF16::from_float(Float::reinterpret_from_int(0x7FC00000)).is_nan())
  // A NaN payload entirely in the dropped half must not become infinity.
  assert_true(BF16::from_float_truncate(Float::reinterpret_from_int(0x7F800001)).is_nan())
  assert_eq(BF16::from_float(0.0).bits(), 0x0000)
  assert_eq(BF16::from_float(-0.0).bits(), 0x8000)
  assert_true(BF16::min_subnormal().is_subnormal())
  assert_true(BF16::min_normal().is_normal())
  assert_true(BF16::max_value().is_finite())
  assert_eq(BF16::precision(), 8)
  assert_eq(BF16::epsilon().bits(), 0x3F00)
}
