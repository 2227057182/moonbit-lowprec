/// Published reference vectors.
///
/// These exist so that another low-precision implementation can check itself
/// against this one without importing the code: the pairs below are the
/// values the exhaustive tests already verified, pinned to decimal form.
/// Every entry is covered by `verify_vectors`, and a test asserts that the
/// table still reproduces in full.
///
/// Decimal literals are written out instead of using exponent notation
/// because this dialect has no exponent-form float literal.

/// binary16 patterns widened to binary64. Chosen to cover subnormal, the
/// smallest normal, every exponent decade up to the maximum finite value.
pub let f16_vectors : Array[(Int, Double)] = [
  (0x0000, 0.0),
  (0x0001, 0.000000059604644775390625),
  (0x03FF, 0.000060975551605224609375),
  (0x0400, 0.00006103515625),
  (0x0800, 0.0001220703125),
  (0x1400, 0.0009765625),
  (0x2000, 0.0078125),
  (0x3000, 0.125),
  (0x3555, 0.333251953125),
  (0x3800, 0.5),
  (0x3C00, 1.0),
  (0x3C01, 1.0009765625),
  (0x3E00, 1.5),
  (0x4000, 2.0),
  (0x4200, 3.0),
  (0x4800, 8.0),
  (0x4C00, 16.0),
  (0x5800, 128.0),
  (0x6800, 2048.0),
  (0x7BFF, 65504.0),
  (0xBC00, -1.0),
  (0xC000, -2.0),
  (0xF800, -32768.0),
  (0xFBFF, -65504.0),
]

/// bfloat16 patterns widened to binary64. The exponent range matches
/// binary32, so the large entries are exactly representable in binary64.
pub let bf16_vectors : Array[(Int, Double)] = [
  (0x0000, 0.0),
  (0x3E00, 0.125),
  (0x3E80, 0.25),
  (0x3F00, 0.5),
  (0x3F80, 1.0),
  (0x3F81, 1.0078125),
  (0x4000, 2.0),
  (0x4049, 3.140625),
  (0x4080, 4.0),
  (0x4800, 131072.0),
  (0x5000, 8589934592.0),
  (0x7F7F, 338953138925153547590470800371487866880.0),
  (0xBF80, -1.0),
  (0xC000, -2.0),
]

/// int8 codes at scale 0.0078125, zero point 0, dequantized value in the
/// second column. Endpoints included, so the full code range is pinned.
pub let quant_vectors : Array[(Int, Double)] = [
  (-128, -1.0),
  (-100, -0.78125),
  (-64, -0.5),
  (-1, -0.0078125),
  (0, 0.0),
  (1, 0.0078125),
  (16, 0.125),
  (32, 0.25),
  (64, 0.5),
  (100, 0.78125),
  (127, 0.9921875),
]

/// Fixed-point raw values with frac 3 in a 16-bit field.
pub let fixed_vectors : Array[(Int, Double)] = [
  (-1024, -128.0),
  (-512, -64.0),
  (-8, -1.0),
  (-4, -0.5),
  (-1, -0.125),
  (0, 0.0),
  (1, 0.125),
  (4, 0.5),
  (8, 1.0),
  (512, 64.0),
  (1023, 127.875),
]

/// Check the whole table against the live implementation.
///
/// Returns `(checked, failed)`; `failed` being zero means the published
/// values still reproduce.
pub fn verify_vectors() -> (Int, Int) {
  let mut checked = 0
  let mut failed = 0
  for i = 0; i < f16_vectors.length(); i = i + 1 {
    let (bits, expected) = f16_vectors[i]
    checked = checked + 1
    if F16::from_bits(bits).to_double() != expected {
      failed = failed + 1
    }
  }
  for i = 0; i < bf16_vectors.length(); i = i + 1 {
    let (bits, expected) = bf16_vectors[i]
    checked = checked + 1
    if BF16::from_bits(bits).to_double() != expected {
      failed = failed + 1
    }
  }
  let qp = QuantParams::int8(0.0078125, 0)
  for i = 0; i < quant_vectors.length(); i = i + 1 {
    let (code, expected) = quant_vectors[i]
    checked = checked + 1
    if dequantize(code, qp) != expected {
      failed = failed + 1
    }
  }
  for i = 0; i < fixed_vectors.length(); i = i + 1 {
    let (raw, expected) = fixed_vectors[i]
    checked = checked + 1
    if Fixed::from_raw(raw, 3, 16).to_double() != expected {
      failed = failed + 1
    }
  }
  (checked, failed)
}
