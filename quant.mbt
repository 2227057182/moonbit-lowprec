/// Affine quantization parameters: `x = (q - zero_point) * scale`.
pub struct QuantParams {
  scale : Double
  zero_point : Int
  qmin : Int
  qmax : Int
}

/// Build arbitrary parameters.
pub fn QuantParams::new(scale : Double, zero_point : Int, qmin : Int, qmax : Int) -> QuantParams {
  { scale, zero_point, qmin, qmax }
}

/// Signed 8-bit range.
pub fn QuantParams::int8(scale : Double, zero_point : Int) -> QuantParams {
  { scale, zero_point, qmin: -128, qmax: 127 }
}

/// Unsigned 8-bit range.
pub fn QuantParams::uint8(scale : Double, zero_point : Int) -> QuantParams {
  { scale, zero_point, qmin: 0, qmax: 255 }
}

/// The real scale.
pub fn QuantParams::scale_value(self : QuantParams) -> Double {
  self.scale
}

/// The zero point, the integer that represents real zero.
pub fn QuantParams::zero(self : QuantParams) -> Int {
  self.zero_point
}

/// Round a Double to an integer under one of the five modes.
///
/// Written here independently of the fixed-point integer path: one rounds a
/// real number, the other rounds an integer after a bit shift, and they are
/// cross-checked against each other in the tests.
fn round_double(x : Double, mode : Rounding) -> Int {
  let f = x.floor()
  let frac = x - f
  match mode {
    TowardZero => if x >= 0.0 {
      f.to_int()
    } else {
      x.ceil().to_int()
    }
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

/// Clamp an integer into the parameter range.
fn clamp_to(q : Int, p : QuantParams) -> Int {
  if q < p.qmin {
    p.qmin
  } else if q > p.qmax {
    p.qmax
  } else {
    q
  }
}

/// Quantize a real value to an integer code.
pub fn quantize(x : Double, p : QuantParams, rounding : Rounding) -> Int {
  let scaled = x / p.scale
  clamp_to(round_double(scaled, rounding) + p.zero_point, p)
}

/// Dequantize a code back to a real value.
pub fn dequantize(q : Int, p : QuantParams) -> Double {
  Double::from_int(q - p.zero_point) * p.scale
}

/// A requantization factor `multiplier / 2^shift`, kept as integers so a
/// re-scaling never passes through a float at runtime.
pub struct RequantScale {
  multiplier : Int
  shift : Int
}

/// The integer multiplier.
pub fn RequantScale::multiplier(self : RequantScale) -> Int {
  self.multiplier
}

/// Bits to shift right after multiplying.
pub fn RequantScale::shift(self : RequantScale) -> Int {
  self.shift
}

/// Derive `scale_in / scale_out` as a multiplier normalized into
/// `[2^30, 2^31)` plus a right shift.
pub fn RequantScale::from_scales(scale_in : Double, scale_out : Double) -> RequantScale {
  if scale_in <= 0.0 || scale_out <= 0.0 {
    abort("quantization scales must be positive")
  }
  let ratio = scale_in / scale_out
  let mut m = ratio
  let mut s = 0
  while m < 1073741824.0 {
    m = m * 2.0
    s = s + 1
  }
  while m >= 2147483648.0 {
    m = m / 2.0
    s = s - 1
  }
  { multiplier: m.round().to_int(), shift: s }
}

/// Round a signed 64-bit integer after dropping `shift` low bits.
///
/// The requantization product `(q - zp) * multiplier` reaches about 2^38,
/// which does not fit the 32-bit `Int`; this is the wider path for it.
fn round_signed64(n : Int64, shift : Int, mode : Rounding) -> Int64 {
  let zero = Int64::from_int(0)
  let one = Int64::from_int(1)
  if shift == 0 {
    return n
  }
  if shift < 0 {
    return n << -shift
  }
  if shift > 62 {
    return zero
  }
  let negative = n < zero
  let mag = if negative {
    -n
  } else {
    n
  }
  let base = mag >> shift
  let rem = mag & ((one << shift) - one)
  let up = if rem == zero {
    false
  } else {
    match mode {
      TowardZero => false
      TowardNegative => negative
      TowardPositive => !negative
      NearestEven => {
        let half = one << (shift - 1)
        rem > half || (rem == half && (base & one) == one)
      }
      NearestAway => rem >= (one << (shift - 1))
    }
  }
  let rounded = if up {
    base + one
  } else {
    base
  }
  if negative {
    -rounded
  } else {
    rounded
  }
}

/// Move a code from one quantization domain to another using only integers.
///
/// The product is formed in 64 bits, rounded once, then clamped. No float
/// appears between the input code and the output code.
pub fn requantize(
  q_in : Int,
  in_p : QuantParams,
  out_p : QuantParams,
  rs : RequantScale,
  rounding : Rounding,
) -> Int {
  let diff = q_in - in_p.zero_point
  let v = Int64::from_int(diff) * Int64::from_int(rs.multiplier)
  let rounded = round_signed64(v, rs.shift, rounding)
  clamp_to(rounded.to_int() + out_p.zero_point, out_p)
}

/// Quantize with one parameter set per channel.
pub fn quantize_channel(
  x : Double,
  params : Array[QuantParams],
  channel : Int,
  rounding : Rounding,
) -> Int {
  quantize(x, params[channel], rounding)
}

/// Dequantize with one parameter set per channel.
pub fn dequantize_channel(q : Int, params : Array[QuantParams], channel : Int) -> Double {
  dequantize(q, params[channel])
}

/// Integer dot product, the accumulation step of a quantized matrix
/// multiply. Accumulated in 64 bits so a long row cannot overflow
/// mid-sum.
pub fn dot_i8(a : Array[Int], b : Array[Int]) -> Int64 {
  let n = if a.length() < b.length() {
    a.length()
  } else {
    b.length()
  }
  let mut acc = Int64::from_int(0)
  for i = 0; i < n; i = i + 1 {
    acc = acc + Int64::from_int(a[i] * b[i])
  }
  acc
}

/// Dequantize one GGML Q4_0 block: 32 weights packed as 16 bytes of two
/// 4-bit codes each, with a binary16 scale.
///
/// Codes are unsigned nibbles in `0..15` offset by 8, so the stored value
/// is `(code - 8) * d`. This is the format GGUF files use for Q4_0 tensors,
/// and it is the point where `F16` meets a real container.
pub fn ggml_q4_0_dequantize(scale : F16, qs : Array[Int]) -> Array[Double] {
  let d = scale.to_float().to_double()
  let out = Array::make(32, 0.0)
  for i = 0; i < 16; i = i + 1 {
    let byte = qs[i]
    let lo = byte & 0x0F
    let hi = (byte >> 4) & 0x0F
    out[i * 2] = Double::from_int(lo - 8) * d
    out[i * 2 + 1] = Double::from_int(hi - 8) * d
  }
  out
}
