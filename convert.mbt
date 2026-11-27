/// Conversions between the low-precision formats that do not go through
/// binary32 or binary64. Each takes the source pattern apart into a sign, an
/// integer significand and a power of two, then rounds once into the target
/// format — the same discipline the arithmetic uses.
///
/// The pair worth stating explicitly is bfloat16 to binary16: within the range
/// both can hold, seven significand bits become ten, so the conversion is exact
/// and no rounding happens at all. Only the exponent range differs, which is
/// where the infinity and zero results come from.

fn bit_length(v : Int64) -> Int {
  let mut n = 0
  let mut x = v
  while x > 0 {
    x = x >> 1
    n = n + 1
  }
  n
}

/// Rounds `v` right by `shift` bits, ties to even. A non-positive shift left
/// shifts, which cannot lose a bit.
fn round_shift_even(v : Int64, shift : Int) -> Int64 {
  if shift <= 0 {
    return v << (0 - shift)
  }
  // Shifting a 64-bit word by 64 or more does not behave as a zero fill, and
  // every value below 2^63 is less than half of a unit at that distance.
  if shift >= 64 {
    return Int64::from_int(0)
  }
  let half = Int64::from_int(1) << (shift - 1)
  let q = v >> shift
  let r = v - (q << shift)
  if r > half {
    q + Int64::from_int(1)
  } else if r == half && (q & Int64::from_int(1)) == Int64::from_int(1) {
    q + Int64::from_int(1)
  } else {
    q
  }
}

/// Encodes a finite value given as `s * 2^et` with `s` an eleven-bit
/// significand into a bfloat16 pattern, rounding the significand to eight bits.
fn encode_bf16(sign : Int, s : Int64, et : Int) -> BF16 {
  let mut e = et
  let mut r = round_shift_even(s, bit_length(s) - 8)
  if r >= Int64::from_int(256) {
    r = Int64::from_int(128)
    e = e + 1
  }
  let eb = e + 127
  let body = (eb << 7) | (r.to_int() - 128)
  BF16::from_bits((sign << 15) | body)
}

/// Encodes a finite value as `s * 2^et` with `s` at most four bits into an
/// e5m2 pattern, rounding the significand to three bits.
fn encode_e5m2(sign : Int, s : Int64, et : Int) -> E5M2 {
  let mut e = et
  let mut r = round_shift_even(s, bit_length(s) - 3)
  if r >= Int64::from_int(8) {
    r = Int64::from_int(4)
    e = e + 1
  }
  let eb = e + 15
  let body = (eb << 2) | (r.to_int() - 4)
  E5M2::from_bits((sign << 7) | body)
}

/// bfloat16 to binary16. Exact for every value binary16 can hold: the seven
/// significand bits fit in ten. Values past 65504 become infinity, values below
/// half the smallest subnormal become zero, and the subnormal band rounds once.
pub fn F16::from_bf16(b : BF16) -> F16 {
  let bits = b.bits()
  let sign = bits >> 15
  let e = (bits >> 7) & 0xFF
  let m = bits & 0x7F
  if e == 0xFF {
    if m == 0 {
      return F16::from_bits((sign << 15) | 0x7C00)
    }
    // The payload fits: seven bits move into the top of ten, quiet bit set.
    return F16::from_bits((sign << 15) | 0x7E00 | (m << 3))
  }
  if e == 0 {
    // The largest bfloat16 subnormal is 127 * 2^-133, far below 2^-25.
    return F16::from_bits(sign << 15)
  }
  let et = e - 127
  if et > 15 {
    return F16::from_bits((sign << 15) | 0x7C00)
  }
  if et >= -14 {
    return F16::from_bits((sign << 15) | ((et + 15) << 10) | (m << 3))
  }
  // Target subnormal: the unit is 2^-24 and the value is (128 + m) * 2^(e-134).
  let s = Int64::from_int(128 + m)
  let shift = 110 - e
  let k = round_shift_even(s, shift).to_int()
  if k == 0 {
    return F16::from_bits(sign << 15)
  }
  F16::from_bits((sign << 15) | k)
}

/// binary16 to bfloat16. Every binary16 value is inside the bfloat16 range, so
/// there is no infinity result; the eleven-bit significand rounds to eight.
pub fn BF16::from_f16(f : F16) -> BF16 {
  let bits = f.bits()
  let sign = bits >> 15
  let e = (bits >> 10) & 0x1F
  let m = bits & 0x3FF
  if e == 0x1F {
    if m == 0 {
      return BF16::from_bits((sign << 15) | 0x7F80)
    }
    return BF16::from_bits((sign << 15) | 0x7F80 | (m >> 3) | 0x40)
  }
  if e == 0 {
    if m == 0 {
      return BF16::from_bits(sign << 15)
    }
    // Normalise the subnormal: value = m * 2^-24.
    let p = bit_length(Int64::from_int(m)) - 1
    let s = Int64::from_int(m) << (10 - p)
    return encode_bf16(sign, s, p - 24)
  }
  encode_bf16(sign, Int64::from_int(1024 + m), e - 15)
}

/// e4m3 to binary16. Exact: four significand bits fit in ten and the range is
/// inside binary16's.
pub fn F16::from_e4m3(x : E4M3) -> F16 {
  let bits = x.bits()
  let sign = bits >> 7
  let e = (bits >> 3) & 0xF
  let m = bits & 7
  if e == 0xF && m == 7 {
    return F16::from_bits((sign << 15) | 0x7E00)
  }
  if e == 0 {
    if m == 0 {
      return F16::from_bits(sign << 15)
    }
    let p = bit_length(Int64::from_int(m)) - 1
    let s = m << (10 - p)
    return F16::from_bits((sign << 15) | ((p - 9 + 15) << 10) | (s - 1024))
  }
  let et = e - 7
  F16::from_bits((sign << 15) | ((et + 15) << 10) | (m << 7))
}

/// e5m2 to binary16. Exact for normals; its three subnormal values land on the
/// binary16 subnormal grid.
pub fn F16::from_e5m2(x : E5M2) -> F16 {
  let bits = x.bits()
  let sign = bits >> 7
  let e = (bits >> 2) & 0x1F
  let m = bits & 3
  if e == 0x1F {
    if m == 0 {
      return F16::from_bits((sign << 15) | 0x7C00)
    }
    return F16::from_bits((sign << 15) | 0x7E00)
  }
  if e == 0 {
    // m * 2^-16 as binary16 subnormal units of 2^-24.
    return F16::from_bits((sign << 15) | (m << 8))
  }
  F16::from_bits((sign << 15) | ((e - 15 + 15) << 10) | (m << 8))
}

/// e4m3 to e5m2. The significand shrinks from four bits to three, so this is the
/// one fp8-to-fp8 direction that rounds.
pub fn E5M2::from_e4m3(x : E4M3) -> E5M2 {
  let bits = x.bits()
  let sign = bits >> 7
  let e = (bits >> 3) & 0xF
  let m = bits & 7
  if e == 0xF && m == 7 {
    return E5M2::from_bits((sign << 7) | 0x7F)
  }
  if e == 0 {
    if m == 0 {
      return E5M2::from_bits(sign << 7)
    }
    let p = bit_length(Int64::from_int(m)) - 1
    return encode_e5m2(sign, Int64::from_int(m) << (2 - p), p - 9)
  }
  encode_e5m2(sign, Int64::from_int(8 + m), e - 7)
}

/// e5m2 to e4m3. The significand grows from three bits to four, so normals are
/// exact; the range shrinks, so anything past 448 saturates and the three
/// subnormals fall below half the smallest e4m3 subnormal and become zero.
pub fn E4M3::from_e5m2(x : E5M2) -> E4M3 {
  let bits = x.bits()
  let sign = bits >> 7
  let e = (bits >> 2) & 0x1F
  let m = bits & 3
  if e == 0x1F {
    if m == 0 {
      return E4M3::from_bits((sign << 7) | 0x7E)
    }
    return E4M3::from_bits((sign << 7) | 0x7F)
  }
  if e == 0 {
    // m * 2^-16 is at most 3 * 2^-16, below half of 2^-9.
    return E4M3::from_bits(sign << 7)
  }
  let et = e - 15
  if et > 8 {
    return E4M3::from_bits((sign << 7) | 0x7E)
  }
  if et >= -6 {
    return E4M3::from_bits((sign << 7) | ((et + 7) << 3) | (m << 1))
  }
  // Target subnormal: the unit is 2^-9 and the value is (4 + m) * 2^(et-2).
  let k = round_shift_even(Int64::from_int(4 + m), 0 - (et + 7)).to_int()
  E4M3::from_bits((sign << 7) | k)
}
