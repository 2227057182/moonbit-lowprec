test "exact signal reports infinite snr" {
  // Every value is exactly representable at scale 0.5, so nothing is lost.
  let values = [0.5, -1.0, 1.5]
  let p = QuantParams::int8(0.5, 0)
  let r = quantization_report(values, p, NearestEven)
  assert_eq(r.count, 3)
  assert_eq(r.max_abs_error, 0.0)
  assert_eq(r.snr_db, double_infinity())
}

test "snr matches a double reference" {
  // 0.3 -> 0.5 (err 0.2), -0.7 -> -0.5 (err 0.2), 1.1 -> 1.0 (err -0.1)
  let values = [0.3, -0.7, 1.1]
  let p = QuantParams::int8(0.5, 0)
  let r = quantization_report(values, p, NearestEven)
  let signal = 0.3 * 0.3 + 0.7 * 0.7 + 1.1 * 1.1
  let noise = 0.2 * 0.2 + 0.2 * 0.2 + 0.1 * 0.1
  let expected = 10.0 * @math.log10(signal / noise)
  assert_eq(r.snr_db - expected < 0.0000001 && expected - r.snr_db < 0.0000001, true)
  assert_eq(r.max_abs_error < 0.2000001 && r.max_abs_error > 0.1999999, true)
  assert_eq(r.mean_abs_error < 0.1667 && r.mean_abs_error > 0.1666, true)
}

test "nearest even never exceeds half a step" {
  // 990 values covering -0.99 .. 0.99, all inside the int8 range at this
  // scale, so the bound is the rounding bound and not a saturation artefact.
  let n = 990
  let values = Array::make(n, 0.0)
  for i = 0; i < n; i = i + 1 {
    values[i] = Double::from_int(i - 495) * 0.002
  }
  let p = QuantParams::int8(0.0078125, 0)
  let r = quantization_report(values, p, NearestEven)
  assert_eq(r.count, n)
  assert_eq(r.max_abs_error <= 0.00390625, true)
}

test "saturation dominates the error at the edge" {
  // 100.0 needs code 200 at this scale; int8 stops at 127, so the clamp must
  // appear in the report rather than hide behind a rounding bound.
  let values = [100.0]
  let p = QuantParams::int8(0.5, 0)
  let r = quantization_report(values, p, NearestEven)
  assert_eq(r.max_abs_error > 30.0, true)
}

test "zero signal and empty signal" {
  let p = QuantParams::int8(0.5, 0)
  let zero = quantization_report([0.0, 0.0], p, NearestEven)
  assert_eq(zero.max_abs_error, 0.0)
  assert_eq(zero.snr_db, double_infinity())
  let empty = quantization_report([], p, NearestEven)
  assert_eq(empty.count, 0)
}

test "toward zero is worse than nearest even" {
  // Rounding toward zero throws away up to a full step, so the report must
  // show a larger maximum error on the same signal.
  let n = 990
  let values = Array::make(n, 0.0)
  for i = 0; i < n; i = i + 1 {
    values[i] = (Double::from_int(i - 495) * 0.002) + 0.001
  }
  let p = QuantParams::int8(0.0078125, 0)
  let nearest = quantization_report(values, p, NearestEven)
  let trunc = quantization_report(values, p, TowardZero)
  assert_eq(trunc.max_abs_error > nearest.max_abs_error, true)
}
