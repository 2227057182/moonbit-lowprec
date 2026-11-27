/// Arithmetic on the two 16-bit float types.
///
/// Each operation is one rounding: the integer core in `fp_arith` combines
/// the significands exactly and rounds once. Nothing here uses binary64, so
/// the results do not depend on the backend's float support.

pub fn F16::add(self : F16, other : F16) -> F16 {
  F16::from_bits(f16_add_bits(self.bits(), other.bits()))
}

pub fn F16::sub(self : F16, other : F16) -> F16 {
  F16::from_bits(f16_sub_bits(self.bits(), other.bits()))
}

pub fn F16::mul(self : F16, other : F16) -> F16 {
  F16::from_bits(f16_mul_bits(self.bits(), other.bits()))
}

pub fn F16::div(self : F16, other : F16) -> F16 {
  F16::from_bits(f16_div_bits(self.bits(), other.bits()))
}

pub fn BF16::add(self : BF16, other : BF16) -> BF16 {
  BF16::from_bits(bf16_add_bits(self.bits(), other.bits()))
}

pub fn BF16::sub(self : BF16, other : BF16) -> BF16 {
  BF16::from_bits(bf16_sub_bits(self.bits(), other.bits()))
}

pub fn BF16::mul(self : BF16, other : BF16) -> BF16 {
  BF16::from_bits(bf16_mul_bits(self.bits(), other.bits()))
}

pub fn BF16::div(self : BF16, other : BF16) -> BF16 {
  BF16::from_bits(bf16_div_bits(self.bits(), other.bits()))
}
