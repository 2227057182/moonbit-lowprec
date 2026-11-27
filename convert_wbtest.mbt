/// Every conversion here is done with integer arithmetic on the patterns, and
/// every test compares it against the binary64 path that has been checked since
/// 0.1.0. The two sides share no code, so a bit that is unpacked in the wrong
/// place shows up immediately.

test "every bfloat16 pattern reaches binary16 as the double path does" {
  let mut checked = 0
  let mut infinities = 0
  let mut flushed = 0
  for i = 0; i < 65536; i = i + 1 {
    let b = BF16::from_bits(i)
    let direct = F16::from_bf16(b)
    if b.is_nan() {
      assert_eq(direct.is_nan(), true)
    } else {
      assert_eq(direct.bits(), F16::from_double(b.to_double()).bits())
      checked = checked + 1
    }
    if direct.is_infinite() {
      infinities = infinities + 1
    }
    if direct.is_zero() && !b.is_zero() {
      flushed = flushed + 1
    }
  }
  assert_eq(checked, 65536 - 254)
  // 2^15 * 2^7 = 65504 is the largest, so exponents 143..254 overflow: 112
  // exponents * 128 significands * 2 signs, plus the two patterns that are
  // already infinity. Everything under 2^-25 flushes: exponents 1..101 give
  // 25856 values, exponent 102 only the two zero-significand patterns, and
  // the 254 subnormals make 26112.
  assert_eq(infinities, 28674)
  assert_eq(flushed, 26112)
}

test "every binary16 pattern reaches bfloat16 as the double path does" {
  let mut checked = 0
  for i = 0; i < 65536; i = i + 1 {
    let f = F16::from_bits(i)
    let direct = BF16::from_f16(f)
    if f.is_nan() {
      assert_eq(direct.is_nan(), true)
    } else {
      assert_eq(direct.bits(), BF16::from_double(f.to_double()).bits())
      checked = checked + 1
    }
  }
  assert_eq(checked, 65536 - 2046)
}

test "bfloat16 to binary16 keeps the order of the finite patterns" {
  let mut previous = -1
  let mut runs = 0
  for i = 0; i <= 0x7F7F; i = i + 1 {
    let b = BF16::from_bits(i)
    if !b.is_nan() {
      let bits = F16::from_bf16(b).bits()
      if previous >= 0 && bits < previous {
        runs = runs + 1
      }
      previous = bits
    }
  }
  assert_eq(runs, 0)
}

test "every e4m3 pattern widens exactly" {
  for i = 0; i < 256; i = i + 1 {
    let x = E4M3::from_bits(i)
    if x.is_nan() {
      assert_eq(F16::from_e4m3(x).is_nan(), true)
    } else {
      assert_eq(
        F16::from_e4m3(x).bits(),
        F16::from_float(x.to_float()).bits(),
      )
    }
  }
}

test "every e5m2 pattern widens exactly" {
  for i = 0; i < 256; i = i + 1 {
    let x = E5M2::from_bits(i)
    if x.is_nan() {
      assert_eq(F16::from_e5m2(x).is_nan(), true)
    } else {
      assert_eq(
        F16::from_e5m2(x).bits(),
        F16::from_float(x.to_float()).bits(),
      )
    }
  }
}

test "e4m3 reaches e5m2 as the float path does" {
  for i = 0; i < 256; i = i + 1 {
    let x = E4M3::from_bits(i)
    let direct = E5M2::from_e4m3(x)
    let reference = E5M2::from_float(x.to_float())
    if x.is_nan() {
      assert_eq(direct.is_nan(), true)
    } else {
      assert_eq(direct.bits(), reference.bits())
    }
  }
}

test "e5m2 reaches e4m3 as the float path does, saturating above 448" {
  let mut saturated = 0
  for i = 0; i < 256; i = i + 1 {
    let x = E5M2::from_bits(i)
    let direct = E4M3::from_e5m2(x)
    let reference = E4M3::from_float(x.to_float())
    if x.is_nan() {
      assert_eq(direct.is_nan(), true)
    } else {
      assert_eq(direct.bits(), reference.bits())
    }
    if !x.is_nan() && direct.bits() == ((x.bits() >> 7 << 7) | 0x7E) {
      saturated = saturated + 1
    }
  }
  // Two infinities, the two patterns that are already 448, and the 56 patterns
  // from 512 up all land on 448.
  assert_eq(saturated, 60)
}
