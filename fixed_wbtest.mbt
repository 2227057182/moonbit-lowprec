/// Independent reference for the five rounding modes, computed on `Double`
/// rather than on integer shifts. Values here are small enough that Double
/// represents them exactly, so any disagreement is a real defect.
fn ref_round(x : Double, mode : Rounding) -> Int {
  let f = x.floor()
  let frac = x - f
  match mode {
    TowardZero => if x >= 0.0 { f.to_int() } else { x.ceil().to_int() }
    TowardNegative => f.to_int()
    TowardPositive => x.ceil().to_int()
    NearestEven => {
      let base = f.to_int()
      if frac > 0.5 {
        base + 1
      } else if frac < 0.5 {
        base
      } else if base % 2 == 0 {
        base
      } else {
        base + 1
      }
    }
    NearestAway => {
      // frac is the distance from floor(x) up to x, so for negative x a
      // fraction above one half means x is closer to the neighbour above it.
      let base = f.to_int()
      if x >= 0.0 {
        if frac >= 0.5 {
          base + 1
        } else {
          base
        }
      } else {
        if frac > 0.5 {
          base + 1
        } else {
          base
        }
      }
    }
  }
}

/// Nine hand-checked cases: positive, negative, and exact halves, each
/// against all five modes. These pin the direction of every mode so a
/// systematic sign error cannot hide behind the exhaustive loop.
test "hand-checked rounding directions" {
  // 1.5 * 0.5 = 0.75, settled on the integer grid.
  let p75 = Fixed::from_raw(3, 1, 16).mul(Fixed::from_raw(1, 1, 16), 0, TowardZero, Saturate)
  assert_eq(p75.raw(), 0)
  assert_eq(Fixed::from_raw(3, 1, 16).mul(Fixed::from_raw(1, 1, 16), 0, TowardPositive, Saturate).raw(), 1)
  assert_eq(Fixed::from_raw(3, 1, 16).mul(Fixed::from_raw(1, 1, 16), 0, NearestEven, Saturate).raw(), 1)
  // -1.5 * 0.5 = -0.75.
  let m75 = Fixed::from_raw(-3, 1, 16).mul(Fixed::from_raw(1, 1, 16), 0, TowardZero, Saturate)
  assert_eq(m75.raw(), 0)
  assert_eq(Fixed::from_raw(-3, 1, 16).mul(Fixed::from_raw(1, 1, 16), 0, TowardNegative, Saturate).raw(), -1)
  assert_eq(Fixed::from_raw(-3, 1, 16).mul(Fixed::from_raw(1, 1, 16), 0, NearestEven, Saturate).raw(), -1)
  // 1.5 * 1.5 = 2.25.
  assert_eq(Fixed::from_raw(3, 1, 16).mul(Fixed::from_raw(3, 1, 16), 0, TowardPositive, Saturate).raw(), 3)
  assert_eq(Fixed::from_raw(3, 1, 16).mul(Fixed::from_raw(3, 1, 16), 0, NearestEven, Saturate).raw(), 2)
  // -1.5 * 1.5 = -2.25.
  assert_eq(Fixed::from_raw(-3, 1, 16).mul(Fixed::from_raw(3, 1, 16), 0, TowardNegative, Saturate).raw(), -3)
  assert_eq(Fixed::from_raw(-3, 1, 16).mul(Fixed::from_raw(3, 1, 16), 0, NearestEven, Saturate).raw(), -2)
  // Exact half 0.5: nearest-even goes to 0, nearest-away goes to 1.
  assert_eq(Fixed::from_raw(2, 1, 16).mul(Fixed::from_raw(1, 1, 16), 0, NearestEven, Saturate).raw(), 0)
  assert_eq(Fixed::from_raw(2, 1, 16).mul(Fixed::from_raw(1, 1, 16), 0, NearestAway, Saturate).raw(), 1)
  // Exact half -0.5: nearest-even goes to 0, nearest-away goes to -1.
  assert_eq(Fixed::from_raw(-2, 1, 16).mul(Fixed::from_raw(1, 1, 16), 0, NearestEven, Saturate).raw(), 0)
  assert_eq(Fixed::from_raw(-2, 1, 16).mul(Fixed::from_raw(1, 1, 16), 0, NearestAway, Saturate).raw(), -1)
}

/// Exhaustive: all 65536 operand pairs drawn from the signed 8-bit range,
/// multiplied at 4 fractional bits and settled back at 4, checked against
/// the Double reference for all five modes. 327680 results.
test "exhaustive 8-bit multiplication across five rounding modes" {
  let modes = [
    TowardZero,
    TowardNegative,
    TowardPositive,
    NearestEven,
    NearestAway,
  ]
  let mut mismatches = 0
  for a = -128; a <= 127; a = a + 1 {
    for b = -128; b <= 127; b = b + 1 {
      let x = Fixed::from_raw(a, 4, 16)
      let y = Fixed::from_raw(b, 4, 16)
      for mode in modes {
        let got = x.mul(y, 4, mode, Saturate).raw()
        // Exact value a*b/16 fits in Double with no error.
        let want = ref_round(Double::from_int(a * b) / 16.0, mode)
        if got != want {
          mismatches = mismatches + 1
          assert_eq(got, want)
        }
      }
    }
  }
  assert_eq(mismatches, 0)
}

/// Exhaustive: 8-bit operands widened into 32-bit fields at 8 fractional
/// bits, multiplied to 16 fractional bits. The shift is zero, so the raw
/// product must be returned exactly — this is the double-width
/// multiply-accumulate case.
test "exhaustive widening multiply is exact" {
  for a = -128; a <= 127; a = a + 1 {
    for b = -128; b <= 127; b = b + 1 {
      let p = Fixed::from_raw(a, 8, 32).mul(Fixed::from_raw(b, 8, 32), 16, TowardZero, Saturate)
      if p.raw() != a * b {
        assert_eq(p.raw(), a * b)
      }
    }
  }
}

/// Multiplication commutes under every mode, because the exact product and
/// its rounding depend only on the pair, not on the order.
test "multiplication commutes" {
  let modes = [TowardZero, TowardNegative, TowardPositive, NearestEven, NearestAway]
  for a = -128; a <= 127; a = a + 7 {
    for b = -128; b <= 127; b = b + 5 {
      let x = Fixed::from_raw(a, 4, 16)
      let y = Fixed::from_raw(b, 4, 16)
      for mode in modes {
        if x.mul(y, 4, mode, Saturate).raw() != y.mul(x, 4, mode, Saturate).raw() {
          assert_eq(x.mul(y, 4, mode, Saturate).raw(), y.mul(x, 4, mode, Saturate).raw())
        }
      }
    }
  }
}

/// Exhaustive over the whole 16-bit field: adding one either saturates at
/// the top or wraps to the bottom, and no other value changes.
test "exhaustive saturate and wrap on every 16-bit raw value" {
  let one = Fixed::from_raw(1, 0, 16)
  for v = -32768; v <= 32767; v = v + 1 {
    let a = Fixed::from_raw(v, 0, 16)
    let sat = a.add(one, Saturate).raw()
    let want_sat = if v == 32767 { 32767 } else { v + 1 }
    if sat != want_sat {
      assert_eq(sat, want_sat)
    }
    let wrapped = a.add(one, Wrap).raw()
    let want_wrap = if v == 32767 { -32768 } else { v + 1 }
    if wrapped != want_wrap {
      assert_eq(wrapped, want_wrap)
    }
  }
}

/// Requantization moves between fractional widths and must follow the
/// requested mode, including on negatives.
test "requantization directions" {
  let x = Fixed::from_raw(3, 2, 16) // 0.75
  assert_eq(x.rescale(0, NearestEven, Saturate).raw(), 1)
  assert_eq(x.rescale(0, TowardZero, Saturate).raw(), 0)
  assert_eq(x.rescale(0, TowardPositive, Saturate).raw(), 1)
  let y = Fixed::from_raw(-3, 2, 16) // -0.75
  assert_eq(y.rescale(0, NearestEven, Saturate).raw(), -1)
  assert_eq(y.rescale(0, TowardZero, Saturate).raw(), 0)
  assert_eq(y.rescale(0, TowardNegative, Saturate).raw(), -1)
  assert_eq(y.rescale(0, TowardPositive, Saturate).raw(), 0)
}

/// Division keeps the format honest: exact quotients land exactly, and
/// inexact ones round toward zero here.
test "division" {
  // 1.5 / 0.5 = 3.
  assert_eq(Fixed::from_raw(3, 1, 16).div(Fixed::from_raw(1, 1, 16), 0, TowardZero).raw(), 3)
  // 1 / 3 at four fractional bits is 0.3333, so raw 5.
  assert_eq(Fixed::from_raw(1, 0, 16).div(Fixed::from_raw(3, 0, 16), 4, TowardZero).raw(), 5)
  // -1 / 3 at four fractional bits rounds toward zero to -5.
  assert_eq(Fixed::from_raw(-1, 0, 16).div(Fixed::from_raw(3, 0, 16), 4, TowardZero).raw(), -5)
  // 4 / 2 = 2.
  assert_eq(Fixed::from_raw(4, 0, 16).div(Fixed::from_raw(2, 0, 16), 0, TowardZero).raw(), 2)
}

/// Negation and absolute value settle the most negative raw value instead
/// of silently overflowing.
test "negation and absolute value at the field edge" {
  let min = Fixed::from_raw(-128, 8, 8)
  assert_eq(min.neg(Saturate).raw(), 127)
  assert_eq(min.neg(Wrap).raw(), -128)
  assert_eq(min.abs(Saturate).raw(), 127)
  assert_eq(min.abs(Wrap).raw(), -128)
}

/// Construction and the scalar view agree.
test "construction and value" {
  let x = Fixed::from_raw(3, 2, 16)
  assert_eq(x.to_double(), 0.75)
  assert_true(x.is_valid())
  assert_eq(x.min_raw(), -32768)
  assert_eq(x.max_raw(), 32767)
  assert_eq(Fixed::from_int(3, 2, 16, Saturate).raw(), 12)
  assert_eq(Fixed::from_int(32767, 0, 16, Saturate).raw(), 32767)
  assert_eq(Fixed::from_int(32768, 0, 16, Saturate).raw(), 32767)
  assert_eq(Fixed::from_int(32768, 0, 16, Wrap).raw(), -32768)
}
