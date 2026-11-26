/// IEEE 754 binary16 (half precision) stored as a 16-bit pattern.
///
/// The value is kept as the raw bit pattern rather than a host float, so
/// conversions are exact and reproducible on every backend.
pub struct F16 {
  bits : Int
}

/// Wrap a 16-bit pattern. Bits above the low 16 are discarded.
pub fn F16::from_bits(bits : Int) -> F16 {
  { bits: bits & 0xFFFF }
}

/// The raw 16-bit pattern, always in `0 ..= 0xFFFF`.
pub fn F16::bits(self : F16) -> Int {
  self.bits
}

/// Sign bit, `0x8000` when negative.
pub fn F16::sign_bit(self : F16) -> Int {
  self.bits & 0x8000
}

/// The 5-bit biased exponent field.
pub fn F16::exponent_field(self : F16) -> Int {
  (self.bits >> 10) & 0x1F
}

/// The 10-bit trailing significand field.
pub fn F16::mantissa_field(self : F16) -> Int {
  self.bits & 0x3FF
}

/// True for any NaN, including signalling and negative NaN.
pub fn F16::is_nan(self : F16) -> Bool {
  self.exponent_field() == 31 && self.mantissa_field() != 0
}

/// True for positive or negative infinity.
pub fn F16::is_infinite(self : F16) -> Bool {
  self.exponent_field() == 31 && self.mantissa_field() == 0
}

/// True for `+0` and `-0`.
pub fn F16::is_zero(self : F16) -> Bool {
  (self.bits & 0x7FFF) == 0
}

/// True for subnormal values, where the exponent field is zero and the
/// significand is not.
pub fn F16::is_subnormal(self : F16) -> Bool {
  self.exponent_field() == 0 && self.mantissa_field() != 0
}

/// True for finite non-zero normal values.
pub fn F16::is_normal(self : F16) -> Bool {
  let e = self.exponent_field()
  e != 0 && e != 31
}

/// True when the sign bit is set, including for negative zero and NaN.
pub fn F16::is_negative(self : F16) -> Bool {
  (self.bits & 0x8000) != 0
}

/// True for finite values, that is neither infinity nor NaN.
pub fn F16::is_finite(self : F16) -> Bool {
  self.exponent_field() != 31
}

/// Widen to `Float` (binary32). Every binary16 value, including subnormals,
/// is representable in binary32, so this never rounds.
pub fn F16::to_float(self : F16) -> Float {
  let s = self.sign_bit()
  let e = self.exponent_field()
  let m = self.mantissa_field()
  let f32 = if e == 31 {
    if m == 0 {
      (s << 16) | 0x7F800000
    } else {
      // Quiet NaN, payload dropped: binary16 payload is a subset of binary32.
      (s << 16) | 0x7FC00000
    }
  } else if e == 0 {
    if m == 0 {
      s << 16
    } else {
      // Subnormal: shift the significand up until the implicit one appears.
      let mut mm = m
      let mut k = 0
      while (mm & 0x400) == 0 {
        mm = mm << 1
        k = k + 1
      }
      // value = 1.f * 2^(-14-k), so the biased exponent is 113 - k.
      // The ten significand bits already sit at f32 positions 22..13.
      (s << 16) | ((113 - k) << 23) | ((mm & 0x3FF) << 13)
    }
  } else {
    // value = 1.f * 2^(e-15), biased exponent becomes e - 15 + 127.
    (s << 16) | ((e + 112) << 23) | (m << 13)
  }
  Float::reinterpret_from_int(f32)
}

/// Round a non-negative integer down by `shift` bits using
/// round-to-nearest, ties-to-even.
fn round_shift(m : Int, shift : Int) -> Int {
  if shift <= 0 {
    return m << -shift
  }
  if shift > 30 {
    return 0
  }
  let base = m >> shift
  let rem = m & ((1 << shift) - 1)
  let half = 1 << (shift - 1)
  if rem > half || (rem == half && (base & 1) == 1) {
    base + 1
  } else {
    base
  }
}

/// Narrow a `Float` (binary32) to binary16 with round-to-nearest,
/// ties-to-even. Overflow produces infinity; tiny values flush to zero or
/// become subnormal, exactly as IEEE 754 requires.
pub fn F16::from_float(x : Float) -> F16 {
  if x.is_nan() {
    return F16::from_bits(0x7E00)
  }
  let b = x.reinterpret_as_int()
  let s = (b >> 16) & 0x8000
  let e32 = (b >> 23) & 0xFF
  let m32 = b & 0x7FFFFF
  if e32 == 0xFF {
    return F16::from_bits(s | 0x7C00)
  }
  if e32 == 0 && m32 == 0 {
    return F16::from_bits(s)
  }
  // Significand with the implicit bit restored for normal binary32 values.
  let m24 = if e32 == 0 {
    m32
  } else {
    m32 | 0x800000
  }
  // Unbiased exponent of the binary32 value.
  let d = if e32 == 0 {
    -126
  } else {
    e32 - 127
  }
  // Biased binary16 exponent before rounding.
  let e16 = d + 15
  if e16 >= 31 {
    return F16::from_bits(s | 0x7C00)
  }
  if e16 <= 0 {
    // Subnormal binary16: value = m16 * 2^-24, so m16 = round(m24 * 2^(d+1)).
    let shift = -d - 1
    let r = round_shift(m24, shift)
    if r >= 1024 {
      // Rounded up into the smallest normal value.
      return F16::from_bits(s | 0x0400)
    }
    return F16::from_bits(s | r)
  }
  // Normal binary16 keeps 11 significant bits, so drop 13 of the 24.
  let r = round_shift(m24, 13)
  if r >= 2048 {
    let ne = e16 + 1
    if ne >= 31 {
      return F16::from_bits(s | 0x7C00)
    }
    return F16::from_bits(s | (ne << 10))
  }
  F16::from_bits(s | (e16 << 10) | (r - 1024))
}

/// Widen through `Float`; exact for every binary16 value.
pub fn F16::to_double(self : F16) -> Double {
  self.to_float().to_double()
}

/// Narrow a `Double` by first converting to binary32. A double that needs
/// more than binary32 precision loses that information at the intermediate
/// step, which is the documented behaviour of this function.
pub fn F16::from_double(x : Double) -> F16 {
  F16::from_float(Float::from_double(x))
}

/// The smallest positive subnormal, `2^-24`.
pub fn min_subnormal() -> F16 {
  F16::from_bits(0x0001)
}

/// The smallest positive normal, `2^-14`.
pub fn min_normal() -> F16 {
  F16::from_bits(0x0400)
}

/// The largest finite value, 65504.
pub fn max_value() -> F16 {
  F16::from_bits(0x7BFF)
}

/// Positive infinity.
pub fn infinity() -> F16 {
  F16::from_bits(0x7C00)
}

/// Quiet NaN with the canonical payload.
pub fn nan() -> F16 {
  F16::from_bits(0x7E00)
}

/// Precision in bits of the significand, including the implicit bit.
pub fn precision() -> Int {
  11
}

/// Machine epsilon, `2^-10`.
pub fn epsilon() -> F16 {
  F16::from_bits(0x1400)
}
