/// Affine round-trip: for every sampled value the reconstructed value must
/// be within half a quantization step of the original.
test "quantize and dequantize stay within half a step" {
  let p = QuantParams::int8(0.05, 0)
  let mut worst = 0.0
  for i = 0; i < 1200; i = i + 1 {
    let x = -6.0 + Double::from_int(i) * 0.01
    let q = quantize(x, p, NearestEven)
    let back = dequantize(q, p)
    let err = if back - x < 0.0 { x - back } else { back - x }
    if err > worst {
      worst = err
    }
  }
  assert_true(worst <= 0.025 + 0.000000001)
}

/// Out-of-range inputs clamp instead of wrapping around.
test "quantize clamps at the range edges" {
  let p = QuantParams::int8(0.05, 0)
  assert_eq(quantize(100.0, p, NearestEven), 127)
  assert_eq(quantize(-100.0, p, NearestEven), -128)
  let u = QuantParams::uint8(0.05, 128)
  assert_eq(quantize(0.0, u, NearestEven), 128)
  assert_eq(quantize(-100.0, u, NearestEven), 0)
  assert_eq(quantize(100.0, u, NearestEven), 255)
}

/// Requantization must agree with an independent Double computation to
/// within one code, and in practice almost always exactly. Both the exact
/// count and the worst difference are asserted.
test "integral requantization matches the double reference" {
  let in_p = QuantParams::int8(0.0078125, 0) // 2^-7
  let out_p = QuantParams::uint8(0.00390625, 128) // 2^-8, zero point 128
  let rs = RequantScale::from_scales(in_p.scale, out_p.scale)
  assert_eq(rs.multiplier() >= 1073741824, true)
  assert_eq(rs.multiplier() > 0, true)
  let mut exact = 0
  let mut worst = 0
  for q = -128; q <= 127; q = q + 1 {
    let got = requantize(q, in_p, out_p, rs, NearestEven)
    let real = dequantize(q, in_p)
    let ideal = real / out_p.scale + Double::from_int(out_p.zero_point)
    let f = ideal.floor()
    let base = f.to_int()
    let frac = ideal - f
    let picked = if frac > 0.5 {
      base + 1
    } else if frac < 0.5 {
      base
    } else if base % 2 == 0 {
      base
    } else {
      base + 1
    }
    let want = if picked < out_p.qmin {
      out_p.qmin
    } else if picked > out_p.qmax {
      out_p.qmax
    } else {
      picked
    }
    let diff = if got > want { got - want } else { want - got }
    if diff == 0 {
      exact = exact + 1
    }
    if diff > worst {
      worst = diff
    }
  }
  assert_eq(worst <= 1, true)
  assert_eq(exact >= 250, true)
}

/// The scale derivation keeps the multiplier inside its documented band for
/// ratios both below and above one.
test "requant scale normalization" {
  let big = RequantScale::from_scales(0.5, 0.125)
  assert_eq(big.multiplier() >= 1073741824, true)
  assert_eq(big.multiplier() > 0, true)
  let small = RequantScale::from_scales(0.001, 1.0)
  assert_eq(small.multiplier() >= 1073741824, true)
  assert_eq(small.multiplier() > 0, true)
}

/// One parameter set per channel, picked by index.
test "per-channel parameters" {
  let params = [
    QuantParams::int8(0.01, 0),
    QuantParams::int8(0.02, -3),
  ]
  assert_eq(quantize_channel(0.1, params, 0, NearestEven), 10)
  assert_eq(quantize_channel(0.1, params, 1, NearestEven), 2)
  assert_eq(dequantize_channel(10, params, 0), 0.1)
}

/// Integer accumulation is exact and matches a hand sum.
test "integer dot product" {
  assert_eq(dot_i8([1, 2, 3], [4, 5, 6]), Int64::from_int(32))
  assert_eq(dot_i8([-128, 127], [127, -128]), Int64::from_int(-32512))
  let zeros = Array::make(4, 0)
  assert_eq(dot_i8(zeros, zeros), Int64::from_int(0))
}

/// GGML Q4_0: 32 weights from 16 packed bytes with a binary16 scale. The
/// reference values are computed by hand from the format definition.
test "gguf q4_0 block dequantization" {
  // scale 0.5 in binary16 is 0x3800.
  let scale = F16::from_bits(0x3800)
  // Byte 0x00 is two codes of zero, 0x88 is two codes of 8, 0xF1 is 1 and 15.
  let qs = Array::make(16, 0)
  qs[0] = 0x00
  qs[1] = 0x88
  qs[2] = 0xF1
  let out = ggml_q4_0_dequantize(scale, qs)
  assert_eq(out.length(), 32)
  assert_eq(out[0], -4.0) // code 0 -> (0 - 8) * 0.5
  assert_eq(out[3], 0.0) // code 8 -> (8 - 8) * 0.5
  assert_eq(out[4], -3.5) // low nibble 1 -> (1 - 8) * 0.5
  assert_eq(out[5], 3.5) // high nibble 15 -> (15 - 8) * 0.5
}

/// Every code in a Q4_0 block round-trips through the documented affine
/// relation, checked against the reconstructed real value.
test "gguf q4_0 covers all sixteen codes" {
  let scale = F16::from_bits(0x3C00) // 1.0
  let qs = Array::make(16, 0)
  for i = 0; i < 16; i = i + 1 {
    // Two codes per byte: low = 2*i mod 16, high = 2*i+1 mod 16.
    qs[i] = (i % 8) + ((i + 8) % 16) * 16
  }
  let out = ggml_q4_0_dequantize(scale, qs)
  for i = 0; i < 16; i = i + 1 {
    let lo = i % 8
    let hi = (i + 8) % 16
    assert_eq(out[i * 2], Double::from_int(lo - 8))
    assert_eq(out[i * 2 + 1], Double::from_int(hi - 8))
  }
}
