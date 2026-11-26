/// How a value whose low bits must be dropped is rounded.
pub enum Rounding {
  /// Drop the extra bits: move toward zero.
  TowardZero
  /// Always move toward negative infinity, also called floor.
  TowardNegative
  /// Always move toward positive infinity, also called ceil.
  TowardPositive
  /// Nearest, with exact halves going to the value whose last kept bit is
  /// even.
  NearestEven
  /// Nearest, with exact halves going away from zero.
  NearestAway
}

/// What happens when a result leaves the representable range.
pub enum Overflow {
  /// Clamp to the nearest representable endpoint.
  Saturate
  /// Keep only the low `width` bits and reinterpret them as signed.
  Wrap
}

/// Parse a rounding mode from its conventional ordinal, `0` being
/// toward zero.
pub fn Rounding::from_int(n : Int) -> Rounding {
  match n {
    1 => TowardNegative
    2 => TowardPositive
    3 => NearestEven
    4 => NearestAway
    _ => TowardZero
  }
}

/// Parse an overflow mode, `0` being saturation.
pub fn Overflow::from_int(n : Int) -> Overflow {
  match n {
    1 => Wrap
    _ => Saturate
  }
}

/// A signed fixed-point value: `raw * 2^-frac` in a `width`-bit field.
///
/// The storage is an `Int`, which is wider than any supported field, so the
/// exact product of two values always fits and no intermediate rounding is
/// forced. `width` is advisory for arithmetic and authoritative when a
/// result is settled into the field.
pub struct Fixed {
  raw : Int
  frac : Int
  width : Int
}

/// Build from a raw integer. No range check is applied; use `is_valid` or
/// let an arithmetic operation settle the value.
pub fn Fixed::from_raw(raw : Int, frac : Int, width : Int) -> Fixed {
  { raw, frac, width }
}

/// The stored integer.
pub fn Fixed::raw(self : Fixed) -> Int {
  self.raw
}

/// Number of fractional bits.
pub fn Fixed::frac(self : Fixed) -> Int {
  self.frac
}

/// Field width in bits, one of 8, 16 or 32 in practice.
pub fn Fixed::width(self : Fixed) -> Int {
  self.width
}

/// Smallest raw value the field can hold.
pub fn Fixed::min_raw(self : Fixed) -> Int {
  -(1 << (self.width - 1))
}

/// Largest raw value the field can hold.
pub fn Fixed::max_raw(self : Fixed) -> Int {
  (1 << (self.width - 1)) - 1
}

/// True when the raw value fits the declared field.
pub fn Fixed::is_valid(self : Fixed) -> Bool {
  self.raw >= self.min_raw() && self.raw <= self.max_raw()
}

/// Value as a `Double`. Exact whenever the raw value fits in 53 bits, which
/// covers every supported field.
pub fn Fixed::to_double(self : Fixed) -> Double {
  let numerator = Double::from_int(self.raw)
  let denominator = Double::from_int(1 << self.frac)
  numerator / denominator
}

/// Clamp, wrap or pass through an integer according to the overflow mode.
fn settle(v : Int, width : Int, mode : Overflow) -> Int {
  let lo = -(1 << (width - 1))
  let hi = (1 << (width - 1)) - 1
  match mode {
    Saturate => {
      if v < lo {
        lo
      } else if v > hi {
        hi
      } else {
        v
      }
    }
    Wrap => {
      let mask = (1 << width) - 1
      let low = v & mask
      if (low >> (width - 1)) == 1 {
        low - (1 << width)
      } else {
        low
      }
    }
  }
}

/// Round a signed integer after dropping `shift` low bits.
///
/// The magnitude is rounded and the sign restored, so a negative input
/// follows exactly the named direction rather than inheriting C's
/// truncation-toward-zero by accident.
fn round_signed(n : Int, shift : Int, mode : Rounding) -> Int {
  if shift == 0 {
    return n
  }
  if shift < 0 {
    return n << -shift
  }
  if shift > 62 {
    return 0
  }
  let negative = n < 0
  let mag = if negative {
    -n
  } else {
    n
  }
  let base = mag >> shift
  let rem = mag & ((1 << shift) - 1)
  let up = if rem == 0 {
    false
  } else {
    match mode {
      TowardZero => false
      TowardNegative => negative
      TowardPositive => !negative
      NearestEven => {
        let half = 1 << (shift - 1)
        rem > half || (rem == half && (base & 1) == 1)
      }
      NearestAway => rem >= (1 << (shift - 1))
    }
  }
  let rounded = if up {
    base + 1
  } else {
    base
  }
  if negative {
    -rounded
  } else {
    rounded
  }
}

/// Build from an integer value, that is `value * 2^frac`, settled into the
/// field by the overflow mode.
pub fn Fixed::from_int(value : Int, frac : Int, width : Int, overflow : Overflow) -> Fixed {
  let shifted = value << frac
  { raw: settle(shifted, width, overflow), frac, width }
}

/// Add two values of the same format.
pub fn Fixed::add(self : Fixed, other : Fixed, overflow : Overflow) -> Fixed {
  if self.frac != other.frac {
    abort("fixed-point add requires equal fractions")
  }
  { raw: settle(self.raw + other.raw, self.width, overflow), frac: self.frac, width: self.width }
}

/// Subtract two values of the same format.
pub fn Fixed::sub(self : Fixed, other : Fixed, overflow : Overflow) -> Fixed {
  if self.frac != other.frac {
    abort("fixed-point sub requires equal fractions")
  }
  { raw: settle(self.raw - other.raw, self.width, overflow), frac: self.frac, width: self.width }
}

/// Negate, settling `-min_raw` according to the overflow mode.
pub fn Fixed::neg(self : Fixed, overflow : Overflow) -> Fixed {
  { raw: settle(-self.raw, self.width, overflow), frac: self.frac, width: self.width }
}

/// Absolute value, settling the most negative input.
pub fn Fixed::abs(self : Fixed, overflow : Overflow) -> Fixed {
  let v = if self.raw < 0 {
    -self.raw
  } else {
    self.raw
  }
  { raw: settle(v, self.width, overflow), frac: self.frac, width: self.width }
}

/// The exact product of the two raw integers.
///
/// With `Int` storage and fields up to 32 bits this is exact; callers doing
/// multi-term accumulation should add these before any rounding.
pub fn Fixed::mul_raw(self : Fixed, other : Fixed) -> Int {
  self.raw * other.raw
}

/// Multiply and settle into `frac_out`.
///
/// The product is exact in `Int`, then rounded once at the end, so the
/// result does not inherit double rounding from an intermediate step.
pub fn Fixed::mul(self : Fixed, other : Fixed, frac_out : Int, rounding : Rounding, overflow : Overflow) -> Fixed {
  if self.width != other.width {
    abort("fixed-point mul requires equal widths")
  }
  let exact = self.mul_raw(other)
  let shift = self.frac + other.frac - frac_out
  let rounded = round_signed(exact, shift, rounding)
  { raw: settle(rounded, self.width, overflow), frac: frac_out, width: self.width }
}

/// Divide and settle into `frac_out`.
///
/// Division by zero aborts; there is no silent infinity in this format.
pub fn Fixed::div(self : Fixed, other : Fixed, frac_out : Int, rounding : Rounding) -> Fixed {
  if other.raw == 0 {
    abort("fixed-point division by zero")
  }
  // value = (a.raw / 2^fa) / (b.raw / 2^fb) = (a.raw * 2^(fb - fa + fo)) / b.raw
  let shift = other.frac - self.frac + frac_out
  let scaled = if shift >= 0 {
    self.raw << shift
  } else {
    round_signed(self.raw, -shift, rounding)
  }
  let quotient = scaled / other.raw
  { raw: settle(quotient, self.width, Overflow::Saturate), frac: frac_out, width: self.width }
}

/// Move to another fractional width, the operation quantization calls
/// requantization.
pub fn Fixed::rescale(self : Fixed, frac_out : Int, rounding : Rounding, overflow : Overflow) -> Fixed {
  let shift = self.frac - frac_out
  let rounded = round_signed(self.raw, shift, rounding)
  { raw: settle(rounded, self.width, overflow), frac: frac_out, width: self.width }
}
