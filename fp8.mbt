/// 8-bit floating point formats used by fp8 inference.
///
/// Two formats are provided, both stored as the raw 8-bit pattern:
///
/// - `E4M3` is the OCP / ONNX `float8_e4m3fn`: one sign bit, four exponent
///   bits with a bias of 7, three significand bits. It has no infinity; the
///   largest finite value is 448, and only `0x7F` and `0xFF` are NaN.
/// - `E5M2` is the OCP / ONNX `float8_e5m2`: one sign bit, five exponent
///   bits with a bias of 15, two significand bits, and IEEE-style infinities
///   and NaNs. The largest finite value is 57344.
pub struct E4M3 {
  bits : Int
}

pub struct E5M2 {
  bits : Int
}

/// The largest finite `E4M3` value, `0x7E` and `0xFE`.
pub let e4m3_max : Double = 448.0

/// The smallest positive `E4M3` value, `0x01`: 2^-9.
pub let e4m3_min_subnormal : Double = 0.001953125

/// The smallest positive normal `E4M3` value, `0x08`: 2^-6.
pub let e4m3_min_normal : Double = 0.015625

/// The largest finite `E5M2` value, `0x7B` and `0xFB`: 57344.
pub let e5m2_max : Double = 57344.0

/// The smallest positive `E5M2` value, `0x01`: 2^-16.
pub let e5m2_min_subnormal : Double = 0.0000152587890625

/// The smallest positive normal `E5M2` value, `0x04`: 2^-14.
pub let e5m2_min_normal : Double = 0.00006103515625

/// Wrap an 8-bit pattern. Bits above the low eight are discarded.
pub fn E4M3::from_bits(bits : Int) -> E4M3 {
  { bits: bits & 0xFF }
}

pub fn E4M3::bits(self : E4M3) -> Int {
  self.bits
}

pub fn E4M3::sign_bit(self : E4M3) -> Int {
  self.bits & 0x80
}

pub fn E4M3::exponent_field(self : E4M3) -> Int {
  (self.bits >> 3) & 0xF
}

pub fn E4M3::mantissa_field(self : E4M3) -> Int {
  self.bits & 0x7
}

/// Only `0x7F` and `0xFF` are NaN: the exponent is all ones and so is the
/// significand. `0x7E` is 448, a finite value, so the test cannot be
/// `exponent all ones`.
pub fn E4M3::is_nan(self : E4M3) -> Bool {
  (self.bits & 0x7F) == 0x7F
}

/// Always false: `E4M3` as used here has no infinity.
pub fn E4M3::is_infinite(self : E4M3) -> Bool {
  let _ = self
  false
}

pub fn E4M3::is_zero(self : E4M3) -> Bool {
  (self.bits & 0x7F) == 0
}

pub fn E4M3::is_subnormal(self : E4M3) -> Bool {
  self.exponent_field() == 0 && self.mantissa_field() != 0
}

pub fn E4M3::is_normal(self : E4M3) -> Bool {
  self.exponent_field() != 0
}

pub fn E4M3::is_finite(self : E4M3) -> Bool {
  !self.is_nan()
}

pub fn E4M3::is_negative(self : E4M3) -> Bool {
  (self.bits & 0x80) != 0
}

/// Widen to `Float`. Every `E4M3` value is exactly representable in binary32,
/// so this never rounds.
pub fn E4M3::to_float(self : E4M3) -> Float {
  if self.is_nan() {
    return Float::reinterpret_from_int(0x7FC00000)
  }
  let sb = self.bits & 0x80
  let e = self.exponent_field()
  let m = self.mantissa_field()
  if e == 0 {
    if m == 0 {
      // Signed zero: keep the sign at bit 31.
      return Float::reinterpret_from_int(sb << 24)
    }
    // m * 2^-9
    let v : Double = Double::from_int(m) * e4m3_min_subnormal
    return Float::from_double(if sb == 0 { v } else { -v })
  }
  // (1 + m/8) * 2^(e-7)
  let v = (1.0 + Double::from_int(m) / 8.0) * pow2d(e - 7)
  Float::from_double(if sb == 0 { v } else { -v })
}

/// Narrow a `Float` with round-to-nearest, ties-to-even, saturating values
/// above 448 to 448 as the ONNX `Cast` to `float8_e4m3fn` does by default.
pub fn E4M3::from_float(x : Float) -> E4M3 {
  e4m3_encode(x, true)
}

/// Same rounding, but a value outside the finite range becomes NaN instead
/// of being clamped. Used by callers that want overflow to stay visible.
pub fn E4M3::from_float_overflow_nan(x : Float) -> E4M3 {
  e4m3_encode(x, false)
}

fn e4m3_encode(x : Float, saturate : Bool) -> E4M3 {
  if x.is_nan() {
    return E4M3::from_bits(0x7F)
  }
  let b = x.reinterpret_as_int()
  let s = (b >> 24) & 0x80
  let e32 = (b >> 23) & 0xFF
  let m32 = b & 0x7FFFFF
  if e32 == 0xFF {
    // Infinity has no pattern here: the caller chose clamp or NaN.
    return E4M3::from_bits(if saturate { s | 0x7E } else { s | 0x7F })
  }
  if e32 == 0 && m32 == 0 {
    return E4M3::from_bits(s)
  }
  let m24 = if e32 == 0 { m32 } else { m32 | 0x800000 }
  let d = if e32 == 0 { -126 } else { e32 - 127 }
  if d >= -6 {
    // Normal target: four significand bits, so twenty are dropped.
    let mut e8 = d + 7
    let mut r = round_shift(m24, 20)
    if r == 16 {
      r = 8
      e8 = e8 + 1
    }
    // r in 8..=14 is 256 through 448; r = 15 would be the NaN pattern.
    if e8 > 15 || (e8 == 15 && r > 14) {
      return E4M3::from_bits(if saturate { s | 0x7E } else { s | 0x7F })
    }
    E4M3::from_bits(s | (e8 << 3) | (r - 8))
  } else {
    // Subnormal target: units of 2^-9, so the value is m24 * 2^(d-23+9).
    let shift = 14 - d
    let k = round_shift(m24, shift)
    if k >= 8 {
      // Rounded up to the smallest normal value.
      E4M3::from_bits(s | 0x08)
    } else {
      E4M3::from_bits(s | k)
    }
  }
}

pub fn E5M2::from_bits(bits : Int) -> E5M2 {
  { bits: bits & 0xFF }
}

pub fn E5M2::bits(self : E5M2) -> Int {
  self.bits
}

pub fn E5M2::sign_bit(self : E5M2) -> Int {
  self.bits & 0x80
}

pub fn E5M2::exponent_field(self : E5M2) -> Int {
  (self.bits >> 2) & 0x1F
}

pub fn E5M2::mantissa_field(self : E5M2) -> Int {
  self.bits & 0x3
}

pub fn E5M2::is_nan(self : E5M2) -> Bool {
  self.exponent_field() == 31 && self.mantissa_field() != 0
}

pub fn E5M2::is_infinite(self : E5M2) -> Bool {
  self.exponent_field() == 31 && self.mantissa_field() == 0
}

pub fn E5M2::is_zero(self : E5M2) -> Bool {
  (self.bits & 0x7F) == 0
}

pub fn E5M2::is_subnormal(self : E5M2) -> Bool {
  self.exponent_field() == 0 && self.mantissa_field() != 0
}

pub fn E5M2::is_normal(self : E5M2) -> Bool {
  let e = self.exponent_field()
  e != 0 && e != 31
}

pub fn E5M2::is_finite(self : E5M2) -> Bool {
  self.exponent_field() != 31
}

pub fn E5M2::is_negative(self : E5M2) -> Bool {
  (self.bits & 0x80) != 0
}

/// Widen to `Float`, exactly for the same reason as `E4M3`.
pub fn E5M2::to_float(self : E5M2) -> Float {
  let sb = self.bits & 0x80
  let e = self.exponent_field()
  let m = self.mantissa_field()
  if e == 31 {
    if m == 0 {
      return Float::reinterpret_from_int((sb << 24) | 0x7F800000)
    }
    return Float::reinterpret_from_int(0x7FC00000)
  }
  if e == 0 {
    if m == 0 {
      return Float::reinterpret_from_int(sb << 24)
    }
    let v = Double::from_int(m) * e5m2_min_subnormal
    return Float::from_double(if sb == 0 { v } else { -v })
  }
  let v = (1.0 + Double::from_int(m) / 4.0) * pow2d(e - 15)
  Float::from_double(if sb == 0 { v } else { -v })
}

/// Narrow a `Float` with round-to-nearest, ties-to-even. Overflow produces
/// infinity, matching IEEE 754 and the `float8_e5m2` definition.
pub fn E5M2::from_float(x : Float) -> E5M2 {
  if x.is_nan() {
    return E5M2::from_bits(0x7E)
  }
  let b = x.reinterpret_as_int()
  let s = (b >> 24) & 0x80
  let e32 = (b >> 23) & 0xFF
  let m32 = b & 0x7FFFFF
  if e32 == 0xFF {
    if m32 == 0 {
      return E5M2::from_bits(s | 0x7C)
    }
    return E5M2::from_bits(0x7E)
  }
  if e32 == 0 && m32 == 0 {
    return E5M2::from_bits(s)
  }
  let m24 = if e32 == 0 { m32 } else { m32 | 0x800000 }
  let d = if e32 == 0 { -126 } else { e32 - 127 }
  if d >= -14 {
    // Three significand bits, so twenty-one are dropped.
    let mut e8 = d + 15
    let mut r = round_shift(m24, 21)
    if r == 8 {
      r = 4
      e8 = e8 + 1
    }
    if e8 >= 31 {
      return E5M2::from_bits(s | 0x7C)
    }
    E5M2::from_bits(s | (e8 << 2) | (r - 4))
  } else {
    // Subnormal target: units of 2^-16.
    let shift = 7 - d
    let k = round_shift(m24, shift)
    if k >= 4 {
      E5M2::from_bits(s | 0x04)
    } else {
      E5M2::from_bits(s | k)
    }
  }
}

/// `2^k` in binary64, for the small exponents these formats need.
fn pow2d(k : Int) -> Double {
  let mut v = 1.0
  if k >= 0 {
    for i = 0; i < k; i = i + 1 {
      v = v * 2.0
    }
  } else {
    for i = 0; i < -k; i = i + 1 {
      v = v / 2.0
    }
  }
  v
}

