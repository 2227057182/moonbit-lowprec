/// bfloat16: one sign bit, eight exponent bits with the same bias as
/// binary32, seven trailing significand bits.
///
/// The pattern is the high half of a binary32, so widening is a shift and
/// no value is lost. Narrowing is the lossy direction and defaults to
/// round-to-nearest, ties-to-even.
pub struct BF16 {
  bits : Int
}

/// Wrap a 16-bit pattern. Bits above the low 16 are discarded.
pub fn BF16::from_bits(bits : Int) -> BF16 {
  { bits: bits & 0xFFFF }
}

/// The raw 16-bit pattern, always in `0 ..= 0xFFFF`.
pub fn BF16::bits(self : BF16) -> Int {
  self.bits
}

/// Sign bit, `0x8000` when negative.
pub fn BF16::sign_bit(self : BF16) -> Int {
  self.bits & 0x8000
}

/// The 8-bit biased exponent field.
pub fn BF16::exponent_field(self : BF16) -> Int {
  (self.bits >> 7) & 0xFF
}

/// The 7-bit trailing significand field.
pub fn BF16::mantissa_field(self : BF16) -> Int {
  self.bits & 0x7F
}

/// True for any NaN.
pub fn BF16::is_nan(self : BF16) -> Bool {
  self.exponent_field() == 0xFF && self.mantissa_field() != 0
}

/// True for positive or negative infinity.
pub fn BF16::is_infinite(self : BF16) -> Bool {
  self.exponent_field() == 0xFF && self.mantissa_field() == 0
}

/// True for `+0` and `-0`.
pub fn BF16::is_zero(self : BF16) -> Bool {
  (self.bits & 0x7FFF) == 0
}

/// True for subnormal values.
pub fn BF16::is_subnormal(self : BF16) -> Bool {
  self.exponent_field() == 0 && self.mantissa_field() != 0
}

/// True for finite non-zero normal values.
pub fn BF16::is_normal(self : BF16) -> Bool {
  let e = self.exponent_field()
  e != 0 && e != 0xFF
}

/// True when the sign bit is set.
pub fn BF16::is_negative(self : BF16) -> Bool {
  (self.bits & 0x8000) != 0
}

/// True for finite values.
pub fn BF16::is_finite(self : BF16) -> Bool {
  self.exponent_field() != 0xFF
}

/// Widen to `Float`. Exact for every bfloat16 value: the pattern occupies
/// the high half of the binary32 and the low half is zero.
pub fn BF16::to_float(self : BF16) -> Float {
  Float::reinterpret_from_int(self.bits << 16)
}

/// Widen to `Double` through `Float`, exact for every bfloat16 value.
pub fn BF16::to_double(self : BF16) -> Double {
  self.to_float().to_double()
}

/// Narrow a `Float` by truncating the low 16 bits, keeping NaN a NaN.
/// This matches the hardware conversions that simply drop the tail.
pub fn BF16::from_float_truncate(x : Float) -> BF16 {
  let b = x.reinterpret_as_int()
  let e32 = (b >> 23) & 0xFF
  let m32 = b & 0x7FFFFF
  let high = b >> 16
  if e32 == 0xFF && m32 != 0 {
    // A NaN whose payload lived in the dropped half would become infinity.
    return BF16::from_bits(high | 0x0040)
  }
  BF16::from_bits(high)
}

/// Narrow a `Float` with round-to-nearest, ties-to-even.
///
/// bfloat16 shares the exponent width of binary32, so finite inputs cannot
/// overflow by magnitude; values above the largest finite bfloat16 still
/// round to infinity because the gap to `2^128` is crossed, exactly as the
/// standard requires.
pub fn BF16::from_float(x : Float) -> BF16 {
  if x.is_nan() {
    return BF16::from_bits(0x7FC0)
  }
  let b = x.reinterpret_as_int()
  let lsb = (b >> 16) & 1
  let out = (b + 0x7FFF + lsb) >> 16
  BF16::from_bits(out)
}

/// Narrow a `Double` by first converting to binary32.
pub fn BF16::from_double(x : Double) -> BF16 {
  BF16::from_float(Float::from_double(x))
}

/// The smallest positive subnormal, `2^-133`.
pub fn BF16::min_subnormal() -> BF16 {
  BF16::from_bits(0x0001)
}

/// The smallest positive normal, `2^-126`.
pub fn BF16::min_normal() -> BF16 {
  BF16::from_bits(0x0080)
}

/// The largest finite value, about 3.3895314e38.
pub fn BF16::max_value() -> BF16 {
  BF16::from_bits(0x7F7F)
}

/// Positive infinity.
pub fn BF16::infinity() -> BF16 {
  BF16::from_bits(0x7F80)
}

/// Quiet NaN with the canonical payload.
pub fn BF16::nan() -> BF16 {
  BF16::from_bits(0x7FC0)
}

/// Precision in bits of the significand, including the implicit bit.
pub fn BF16::precision() -> Int {
  8
}

/// Machine epsilon, `2^-7`.
pub fn BF16::epsilon() -> BF16 {
  BF16::from_bits(0x3F00)
}
