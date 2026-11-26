fn usage() -> Unit {
  println("moonbit-lowprec inspector")
  println("")
  println("  f16 <bits>              decode a binary16 pattern (decimal or 0x hex)")
  println("  bf16 <bits>             decode a bfloat16 pattern")
  println("  fixed <raw> <frac> <width>  show a fixed-point value")
  println("  quant <value> <scale> <zp>   quantize then dequantize with int8")
  println("  requant <q> <in_scale> <in_zp> <out_scale> <out_zp>")
  println("  q4_0 <scale_bits> <byte>...  dequantize a GGML Q4_0 block")
  println("  report                  compare four rounding modes on a sine signal")
  println("  verify                  check the published reference vectors")
  println("  selftest                run the built-in checks")
}

fn digit_value(code : Int) -> Int {
  if code >= 48 && code <= 57 {
    code - 48
  } else if code >= 97 && code <= 102 {
    code - 87
  } else if code >= 65 && code <= 70 {
    code - 55
  } else {
    abort("not a hexadecimal digit")
  }
}

/// Accepts plain decimal and `0x` prefixed hex.
fn parse_bits(s : String) -> Int {
  let mut i = 0
  let mut base = 10
  if s.length() >= 2 && s[0].to_int() == 48 && (s[1].to_int() == 120 || s[1].to_int() == 88) {
    base = 16
    i = 2
  }
  let mut v = 0
  while i < s.length() {
    let d = digit_value(s[i].to_int())
    v = v * base + d
    i = i + 1
  }
  v
}

fn parse_int(s : String) -> Int {
  let mut i = 0
  let mut negative = false
  if s.length() > 0 && s[0].to_int() == 45 {
    negative = true
    i = 1
  }
  let mut v = 0
  while i < s.length() {
    let code = s[i].to_int()
    if code < 48 || code > 57 {
      abort("not a decimal digit")
    }
    v = v * 10 + (code - 48)
    i = i + 1
  }
  if negative {
    -v
  } else {
    v
  }
}

fn parse_double(s : String) -> Double {
  let mut i = 0
  let mut negative = false
  if s.length() > 0 && s[0].to_int() == 45 {
    negative = true
    i = 1
  }
  let mut digits = 0
  let mut frac_digits = 0
  let mut seen_dot = false
  while i < s.length() {
    let code = s[i].to_int()
    if code == 46 {
      seen_dot = true
    } else if code >= 48 && code <= 57 {
      digits = digits * 10 + (code - 48)
      if seen_dot {
        frac_digits = frac_digits + 1
      }
    } else {
      abort("not a decimal number")
    }
    i = i + 1
  }
  let mut denom = 1
  for k = 0; k < frac_digits; k = k + 1 {
    denom = denom * 10
  }
  let value = Double::from_int(digits) / Double::from_int(denom)
  if negative {
    -value
  } else {
    value
  }
}

fn class_of_f16(v : @lib.F16) -> String {
  if v.is_nan() {
    "nan"
  } else if v.is_infinite() {
    "infinity"
  } else if v.is_zero() {
    "zero"
  } else if v.is_subnormal() {
    "subnormal"
  } else {
    "normal"
  }
}

fn class_of_bf16(v : @lib.BF16) -> String {
  if v.is_nan() {
    "nan"
  } else if v.is_infinite() {
    "infinity"
  } else if v.is_zero() {
    "zero"
  } else if v.is_subnormal() {
    "subnormal"
  } else {
    "normal"
  }
}

fn show_f16(bits : Int) -> Unit {
  let v = @lib.F16::from_bits(bits)
  println(
    "bits=" + v.bits().to_string() +
    " sign=" + v.sign_bit().to_string() +
    " exp=" + v.exponent_field().to_string() +
    " mant=" + v.mantissa_field().to_string() +
    " class=" + class_of_f16(v) +
    " value=" + v.to_float().to_string()
  )
}

fn show_bf16(bits : Int) -> Unit {
  let v = @lib.BF16::from_bits(bits)
  println(
    "bits=" + v.bits().to_string() +
    " sign=" + v.sign_bit().to_string() +
    " exp=" + v.exponent_field().to_string() +
    " mant=" + v.mantissa_field().to_string() +
    " class=" + class_of_bf16(v) +
    " value=" + v.to_float().to_string()
  )
}

fn show_fixed(raw : Int, frac : Int, width : Int) -> Unit {
  let v = @lib.Fixed::from_raw(raw, frac, width)
  println(
    "raw=" + v.raw().to_string() +
    " frac=" + v.frac().to_string() +
    " width=" + v.width().to_string() +
    " valid=" + v.is_valid().to_string() +
    " value=" + v.to_double().to_string()
  )
}

fn show_quant(x : Double, scale : Double, zp : Int) -> Unit {
  let p = @lib.QuantParams::int8(scale, zp)
  let q = @lib.quantize(x, p, @lib.NearestEven)
  let back = @lib.dequantize(q, p)
  println("code=" + q.to_string() + " value=" + back.to_string())
}

fn show_requant(q : Int, in_scale : Double, in_zp : Int, out_scale : Double, out_zp : Int) -> Unit {
  let in_p = @lib.QuantParams::int8(in_scale, in_zp)
  let out_p = @lib.QuantParams::uint8(out_scale, out_zp)
  let rs = @lib.RequantScale::from_scales(in_scale, out_scale)
  let code = @lib.requantize(q, in_p, out_p, rs, @lib.NearestEven)
  println(
    "multiplier=" + rs.multiplier().to_string() +
    " shift=" + rs.shift().to_string() +
    " code=" + code.to_string() +
    " value=" + @lib.dequantize(code, out_p).to_string()
  )
}

fn show_q4_0(scale_bits : Int, bytes : Array[Int]) -> Unit {
  let scale = @lib.F16::from_bits(scale_bits)
  let values = @lib.ggml_q4_0_dequantize(scale, bytes)
  let mut text = ""
  for i = 0; i < values.length(); i = i + 1 {
    if i > 0 {
      text = text + " "
    }
    text = text + values[i].to_string()
  }
  println(text)
}

fn demo_report() -> Unit {
  // A sine sweep with amplitude 1: the usual shape of a normalized weight
  // or activation, quantized at int8 with a step of 1/128.
  let n = 256
  let values = Array::make(n, 0.0)
  for i = 0; i < n; i = i + 1 {
    values[i] = 0.98 * @math.sin(
      2.0 * 3.141592653589793 * Double::from_int(i) / Double::from_int(n),
    )
  }
  let p = @lib.QuantParams::int8(0.0078125, 0)
  println("signal: 256 sine samples, amplitude 0.98 (inside int8 at this scale)")
  println("int8, scale 0.0078125; the clamp is therefore not the limiting error")
  let modes : Array[@lib.Rounding] = [
    @lib.NearestEven,
    @lib.TowardZero,
    @lib.TowardNegative,
    @lib.TowardPositive,
  ]
  let names = ["nearest-even    ", "toward-zero     ", "toward-negative ", "toward-positive "]
  for i = 0; i < modes.length(); i = i + 1 {
    let r = @lib.quantization_report(values, p, modes[i])
    println(
      names[i] +
      " snr=" + r.snr_db.to_string() +
      " dB  max=" + r.max_abs_error.to_string() +
      "  mean=" + r.mean_abs_error.to_string(),
    )
  }
}

fn run_verify() -> Unit {
  let (checked, failed) = @lib.verify_vectors()
  println(
    "vectors " + (checked - failed).to_string() + "/" + checked.to_string() +
    " reproduce" + (if failed == 0 { "" } else { ", " + failed.to_string() + " differ" }),
  )
}

fn selftest() -> Unit {
  let mut passed = 0
  let mut failed = 0
  let cases = [
    @lib.F16::from_float(1.0).bits() == 0x3C00,
    @lib.F16::from_bits(0x3C00).to_float() == 1.0,
    @lib.BF16::from_float(1.0).bits() == 0x3F80,
    @lib.F16::from_float(0.1).bits() != 0,
    @lib.Fixed::from_raw(3, 1, 16).mul(@lib.Fixed::from_raw(3, 1, 16), 0, @lib.NearestEven, @lib.Saturate).raw() == 2,
    @lib.dequantize(@lib.quantize(1.25, @lib.QuantParams::int8(0.05, 0), @lib.NearestEven), @lib.QuantParams::int8(0.05, 0)) == 1.25,
    @lib.ggml_q4_0_dequantize(@lib.F16::from_bits(0x3800), Array::make(16, 0))[0] == -4.0,
  ]
  for i = 0; i < cases.length(); i = i + 1 {
    if cases[i] {
      passed = passed + 1
    } else {
      failed = failed + 1
    }
  }
  println("selftest " + passed.to_string() + "/" + (passed + failed).to_string() + " passed")
}

fn main {
  let args = @env.args()
  if args.length() < 2 {
    usage()
    return
  }
  let cmd = args[1]
  if cmd == "f16" && args.length() >= 3 {
    show_f16(parse_bits(args[2]))
  } else if cmd == "bf16" && args.length() >= 3 {
    show_bf16(parse_bits(args[2]))
  } else if cmd == "fixed" && args.length() >= 5 {
    show_fixed(parse_int(args[2]), parse_int(args[3]), parse_int(args[4]))
  } else if cmd == "quant" && args.length() >= 5 {
    show_quant(parse_double(args[2]), parse_double(args[3]), parse_int(args[4]))
  } else if cmd == "requant" && args.length() >= 7 {
    show_requant(
      parse_int(args[2]),
      parse_double(args[3]),
      parse_int(args[4]),
      parse_double(args[5]),
      parse_int(args[6]),
    )
  } else if cmd == "q4_0" && args.length() >= 18 {
    let scale = parse_bits(args[2])
    let bytes = Array::make(16, 0)
    for i = 0; i < 16; i = i + 1 {
      bytes[i] = parse_bits(args[3 + i])
    }
    show_q4_0(scale, bytes)
  } else if cmd == "report" {
    demo_report()
  } else if cmd == "verify" {
    run_verify()
  } else if cmd == "selftest" {
    selftest()
  } else {
    usage()
  }
}
