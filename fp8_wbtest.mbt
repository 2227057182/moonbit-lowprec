/// Checks the two 8-bit formats against `fp8_vectors.mbt`, whose 256 rows per
/// format were decoded with exact rational arithmetic by an independent
/// implementation, plus the rounding properties the table cannot show.

test "every e4m3 pattern agrees with the rational table" {
  for b = 0; b < 256; b = b + 1 {
    let row = e4m3_rows[b]
    assert_eq(row.bits, b)
    let v = E4M3::from_bits(b)
    if row.special == 1 {
      assert_eq(v.is_nan(), true)
    } else {
      assert_eq(v.is_nan(), false)
      assert_eq(v.to_float().to_double(), row.value)
    }
  }
}

test "every e5m2 pattern agrees with the rational table" {
  for b = 0; b < 256; b = b + 1 {
    let row = e5m2_rows[b]
    assert_eq(row.bits, b)
    let v = E5M2::from_bits(b)
    if row.special == 1 {
      assert_eq(v.is_nan(), true)
    } else if row.special == 2 {
      assert_eq(v.is_infinite(), true)
    } else {
      assert_eq(v.is_nan(), false)
      assert_eq(v.to_float().to_double(), row.value)
    }
  }
}

test "narrowing reproduces every finite pattern" {
  // Both directions of the conversion must be exact for values the format
  // can hold, including subnormals and signed zero.
  for b = 0; b < 256; b = b + 1 {
    let v = E4M3::from_bits(b)
    if !v.is_nan() {
      assert_eq(E4M3::from_float(v.to_float()).bits(), b)
    }
  }
  for b = 0; b < 256; b = b + 1 {
    let v = E5M2::from_bits(b)
    if !v.is_nan() {
      assert_eq(E5M2::from_float(v.to_float()).bits(), b)
    }
  }
}

test "positive values increase with the pattern" {
  let mut prev = -1.0
  for b = 0; b < 0x7F; b = b + 1 {
    let v = E4M3::from_bits(b).to_float().to_double()
    assert_eq(v > prev, true)
    prev = v
  }
  prev = -1.0
  for b = 0; b < 0x7C; b = b + 1 {
    let v = E5M2::from_bits(b).to_float().to_double()
    assert_eq(v > prev, true)
    prev = v
  }
}

test "e4m3 saturates or reports overflow, and has no infinity" {
  // 448 is the largest value; the next representable step, 512, does not
  // exist, so the rounding boundary sits at 480.
  assert_eq(E4M3::from_float(448.0).bits(), 0x7E)
  assert_eq(E4M3::from_float(479.0).bits(), 0x7E)
  assert_eq(E4M3::from_float(500.0).bits(), 0x7E)
  assert_eq(E4M3::from_float(-500.0).bits(), 0xFE)
  assert_eq(E4M3::from_float_overflow_nan(500.0).bits(), 0x7F)
  assert_eq(E4M3::from_float_overflow_nan(-500.0).bits(), 0xFF)
  // An infinity input cannot be stored, so it takes the same two paths.
  assert_eq(E4M3::from_float(1.0 / 0.0).bits(), 0x7E)
  assert_eq(E4M3::from_float_overflow_nan(1.0 / 0.0).bits(), 0x7F)
  assert_eq(E4M3::from_bits(0x7E).is_infinite(), false)
}

test "e4m3 subnormal rounding and the tie at 1.5 ulp" {
  assert_eq(E4M3::from_float(0.0005).bits(), 0x00)
  assert_eq(E4M3::from_float(0.001).bits(), 0x01)
  assert_eq(E4M3::from_float(0.001953125).bits(), 0x01)
  // 1.5 * 2^-9 is exactly between 0x01 and 0x02, so ties-to-even picks the
  // even pattern.
  assert_eq(E4M3::from_float(0.0029296875).bits(), 0x02)
  assert_eq(E4M3::from_float(0.015625).bits(), 0x08)
}

test "e5m2 keeps ieee infinities" {
  assert_eq(E5M2::from_float(57344.0).bits(), 0x7B)
  assert_eq(E5M2::from_float(60000.0).bits(), 0x7B)
  // 61440 is halfway to 65536, which is out of range.
  assert_eq(E5M2::from_float(62000.0).bits(), 0x7C)
  assert_eq(E5M2::from_float(-62000.0).bits(), 0xFC)
  assert_eq(E5M2::from_float(1.0 / 0.0).bits(), 0x7C)
  assert_eq(E5M2::from_float(0.0000152587890625).bits(), 0x01)
  assert_eq(E5M2::from_float(0.0001).bits(), 0x07)
}

test "nan patterns are exactly the ones the formats define" {
  let mut nan_count = 0
  for b = 0; b < 256; b = b + 1 {
    if E4M3::from_bits(b).is_nan() {
      nan_count = nan_count + 1
    }
  }
  assert_eq(nan_count, 2)
  nan_count = 0
  for b = 0; b < 256; b = b + 1 {
    if E5M2::from_bits(b).is_nan() {
      nan_count = nan_count + 1
    }
  }
  assert_eq(nan_count, 6)
  // Every NaN narrows to the quiet pattern, never to a number.
  assert_eq(E4M3::from_float(E4M3::from_bits(0x7F).to_float()).is_nan(), true)
}

test "published constants match their patterns" {
  assert_eq(e4m3_max, 448.0)
  assert_eq(e5m2_max, 57344.0)
  assert_eq(E4M3::from_bits(0x7E).to_float().to_double(), e4m3_max)
  assert_eq(E5M2::from_bits(0x7B).to_float().to_double(), e5m2_max)
  assert_eq(E4M3::from_bits(0x01).to_float().to_double(), e4m3_min_subnormal)
  assert_eq(E4M3::from_bits(0x08).to_float().to_double(), e4m3_min_normal)
  assert_eq(E5M2::from_bits(0x01).to_float().to_double(), e5m2_min_subnormal)
  assert_eq(E5M2::from_bits(0x04).to_float().to_double(), e5m2_min_normal)
}
