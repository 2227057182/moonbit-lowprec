/// Size-parametric IEEE 754 binary arithmetic for the small formats.
///
/// binary16 and bfloat16 differ only in field widths, so one integer core
/// serves both: decode each operand into `sign, m, e` with `value =
/// m * 2^e`, combine the significands exactly in 64 bits, and round once at
/// the end. No binary64 instruction is used, so the result is the same on
/// every backend and would still work on a target without a float unit.
///
/// Rounding is to nearest, ties to even, applied at the single place where
/// precision is lost. The rounding therefore sees every discarded bit —
/// there is no intermediate rounding to be fooled by.

priv struct FpParts {
  sign : Int
  m : Int64
  e : Int
  /// 0 finite, 1 infinity, 2 NaN, 3 zero.
  kind : Int
}

fn bit_length64(v : Int64) -> Int {
  let mut x = v
  let mut n = 0
  while x > Int64::from_int(0) {
    x = x >> 1
    n = n + 1
  }
  n
}

/// Shifts right by `shift` with round-to-nearest, ties to even. A negative
/// shift becomes a left shift, which is exact by construction: callers only
/// pass negative shifts when the value is known to fit.
fn shr_round64(m : Int64, shift : Int) -> Int64 {
  if shift <= 0 {
    m << -shift
  } else if shift > 62 {
    Int64::from_int(0)
  } else {
    let one = Int64::from_int(1)
    let q = m >> shift
    let r = m & ((one << shift) - one)
    let half = one << (shift - 1)
    if r > half || (r == half && ((q & one) == one)) {
      q + one
    } else {
      q
    }
  }
}

/// Encodes a positive magnitude `m * 2^e` into the exponent and mantissa
/// fields, returning the low bits without the sign. Out-of-range values
/// return infinity, and values below the subnormal range return zero or the
/// smallest subnormal depending on the rounding.
fn fp_encode(m : Int64, e : Int, mb : Int, eb : Int, bias : Int) -> Int {
  if m == Int64::from_int(0) {
    return 0
  }
  let inf = ((1 << eb) - 1) << mb
  let bitlen = bit_length64(m)
  let value_exp = bitlen - 1 + e
  let min_normal = 1 - bias
  if value_exp < min_normal {
    // Subnormal: the only exponent available is the minimum one, so the
    // significand is scaled to it and rounded there.
    let target_e = 1 - bias - mb
    let shift = target_e - e
    let mm = shr_round64(m, shift)
    let hidden = Int64::from_int(1 << mb)
    if mm >= hidden {
      // Rounded up into the smallest normal.
      1 << mb
    } else {
      mm.to_int()
    }
  } else {
    let shift = bitlen - (mb + 1)
    let mut mm = shr_round64(m, shift)
    let mut ee = e + shift
    let hidden = Int64::from_int(1 << mb)
    if mm >= (hidden << 1) {
      // The round carried into a new bit; the low bit is necessarily zero.
      mm = mm >> 1
      ee = ee + 1
    }
    let exp = ee + bias + mb
    if exp >= (1 << eb) - 1 {
      return inf
    }
    if exp <= 0 {
      return inf
    }
    (exp << mb) | (mm.to_int() - (1 << mb))
  }
}

fn fp_parts(bits : Int, mb : Int, eb : Int, bias : Int) -> FpParts {
  let total = 1 + eb + mb
  let sign = (bits >> (total - 1)) & 1
  let max_exp = (1 << eb) - 1
  let exp = (bits >> mb) & max_exp
  let mant = bits & ((1 << mb) - 1)
  if exp == max_exp {
    if mant == 0 {
      { sign, m: Int64::from_int(0), e: 0, kind: 1 }
    } else {
      { sign, m: Int64::from_int(0), e: 0, kind: 2 }
    }
  } else if exp == 0 {
    if mant == 0 {
      { sign, m: Int64::from_int(0), e: 1 - bias - mb, kind: 3 }
    } else {
      { sign, m: Int64::from_int(mant), e: 1 - bias - mb, kind: 0 }
    }
  } else {
    let m = Int64::from_int((1 << mb) | mant)
    { sign, m, e: exp - bias - mb, kind: 0 }
  }
}

fn fp_quiet_nan(mb : Int, eb : Int) -> Int {
  (((1 << eb) - 1) << mb) | (1 << (mb - 1))
}

fn fp_add_bits(a : Int, b : Int, mb : Int, eb : Int, bias : Int) -> Int {
  let total = 1 + eb + mb
  let pa = fp_parts(a, mb, eb, bias)
  let pb = fp_parts(b, mb, eb, bias)
  if pa.kind == 2 || pb.kind == 2 {
    return fp_quiet_nan(mb, eb)
  }
  if pa.kind == 1 && pb.kind == 1 {
    if pa.sign == pb.sign {
      return (((1 << eb) - 1) << mb) | (pa.sign << (total - 1))
    }
    return fp_quiet_nan(mb, eb)
  }
  if pa.kind == 1 {
    return a
  }
  if pb.kind == 1 {
    return b
  }
  // bfloat16 spans exponents far wider than 64 bits can align: the smallest
  // subnormal is 2^-133 while the largest is near 2^128. When the gap exceeds
  // fifty bits the smaller operand is below half an ulp of the larger one, so
  // the sum rounds back to the larger operand unchanged. binary16 never
  // reaches that gap (its whole range fits in about forty bits), so this only
  // fires for bfloat16.
  let gap = if pa.e > pb.e { pa.e - pb.e } else { pb.e - pa.e }
  if gap > 50 {
    if pa.e >= pb.e {
      return a
    } else {
      return b
    }
  }
  // Align both significands on the smaller exponent. The addition below is
  // exact, so the single rounding at encode time is the only rounding.
  let e = if pa.e < pb.e { pa.e } else { pb.e }
  let mut va = pa.m << (pa.e - e)
  if pa.sign == 1 {
    va = -va
  }
  let mut vb = pb.m << (pb.e - e)
  if pb.sign == 1 {
    vb = -vb
  }
  let sum = va + vb
  if sum == Int64::from_int(0) {
    // (-0) + (-0) is -0; every other zero result is +0 under round to
    // nearest, including x + (-x).
    if va == Int64::from_int(0) && vb == Int64::from_int(0) && pa.sign == pb.sign {
      return pa.sign << (total - 1)
    }
    return 0
  }
  let sign = if sum < Int64::from_int(0) { 1 } else { 0 }
  let magnitude = if sum < Int64::from_int(0) { -sum } else { sum }
  let bits = fp_encode(magnitude, e, mb, eb, bias)
  (sign << (total - 1)) | bits
}

fn fp_sub_bits(a : Int, b : Int, mb : Int, eb : Int, bias : Int) -> Int {
  let total = 1 + eb + mb
  fp_add_bits(a, b ^ (1 << (total - 1)), mb, eb, bias)
}

fn fp_mul_bits(a : Int, b : Int, mb : Int, eb : Int, bias : Int) -> Int {
  let total = 1 + eb + mb
  let inf = ((1 << eb) - 1) << mb
  let pa = fp_parts(a, mb, eb, bias)
  let pb = fp_parts(b, mb, eb, bias)
  let sign = pa.sign ^ pb.sign
  if pa.kind == 2 || pb.kind == 2 {
    return fp_quiet_nan(mb, eb)
  }
  if pa.kind == 1 || pb.kind == 1 {
    if pa.kind == 3 || pb.kind == 3 {
      // infinity times zero
      return fp_quiet_nan(mb, eb)
    }
    return inf | (sign << (total - 1))
  }
  let m = pa.m * pb.m
  let e = pa.e + pb.e
  if m == Int64::from_int(0) {
    return sign << (total - 1)
  }
  let bits = fp_encode(m, e, mb, eb, bias)
  (sign << (total - 1)) | bits
}

fn fp_div_bits(a : Int, b : Int, mb : Int, eb : Int, bias : Int) -> Int {
  let total = 1 + eb + mb
  let inf = ((1 << eb) - 1) << mb
  let pa = fp_parts(a, mb, eb, bias)
  let pb = fp_parts(b, mb, eb, bias)
  let sign = pa.sign ^ pb.sign
  if pa.kind == 2 || pb.kind == 2 {
    return fp_quiet_nan(mb, eb)
  }
  if pa.kind == 1 && pb.kind == 1 {
    return fp_quiet_nan(mb, eb)
  }
  if pa.kind == 1 {
    return inf | (sign << (total - 1))
  }
  if pb.kind == 1 {
    return sign << (total - 1)
  }
  if pb.kind == 3 {
    if pa.kind == 3 {
      return fp_quiet_nan(mb, eb)
    }
    return inf | (sign << (total - 1))
  }
  if pa.kind == 3 {
    return sign << (total - 1)
  }
  // Long division with a 24-bit head: the quotient has at least 23 significant
  // bits, far more than the format needs, and the remainder is carried as a
  // sticky bit in the lowest position so the final rounding cannot miss it.
  let num = pa.m << 24
  let den = pb.m
  let q = num / den
  let r = num % den
  let sticky = if r != Int64::from_int(0) { Int64::from_int(1) } else { Int64::from_int(0) }
  let qq = (q << 1) | sticky
  let e = pa.e - pb.e - 25
  let bits = fp_encode(qq, e, mb, eb, bias)
  (sign << (total - 1)) | bits
}

/// binary16 arithmetic: five exponent bits, ten mantissa bits.
pub fn f16_add_bits(a : Int, b : Int) -> Int {
  fp_add_bits(a, b, 10, 5, 15)
}

pub fn f16_sub_bits(a : Int, b : Int) -> Int {
  fp_sub_bits(a, b, 10, 5, 15)
}

pub fn f16_mul_bits(a : Int, b : Int) -> Int {
  fp_mul_bits(a, b, 10, 5, 15)
}

pub fn f16_div_bits(a : Int, b : Int) -> Int {
  fp_div_bits(a, b, 10, 5, 15)
}

/// bfloat16 arithmetic: eight exponent bits, seven mantissa bits.
pub fn bf16_add_bits(a : Int, b : Int) -> Int {
  fp_add_bits(a, b, 7, 8, 127)
}

pub fn bf16_sub_bits(a : Int, b : Int) -> Int {
  fp_sub_bits(a, b, 7, 8, 127)
}

pub fn bf16_mul_bits(a : Int, b : Int) -> Int {
  fp_mul_bits(a, b, 7, 8, 127)
}

pub fn bf16_div_bits(a : Int, b : Int) -> Int {
  fp_div_bits(a, b, 7, 8, 127)
}
