/// Aggregated error statistics for one quantization configuration applied
/// to a signal.
pub struct QuantReport {
  max_abs_error : Double
  mean_abs_error : Double
  /// Signal-to-quantization-noise ratio in decibels. Positive infinity
  /// means the configuration reproduced the signal exactly.
  snr_db : Double
  count : Int
}

/// Positive infinity as a `Double`. Written as a division so the literal
/// grammar is irrelevant.
pub fn double_infinity() -> Double {
  let zero = Double::from_int(0)
  1.0 / zero
}

/// Measure what a configuration loses on a given signal.
///
/// `snr_db` is `10 * log10(energy(signal) / energy(error))`, the standard
/// figure used when comparing quantization schemes. When the error is
/// exactly zero it reports positive infinity rather than a large sentinel.
pub fn quantization_report(
  values : Array[Double],
  params : QuantParams,
  mode : Rounding,
) -> QuantReport {
  let n = values.length()
  if n == 0 {
    return { max_abs_error: 0.0, mean_abs_error: 0.0, snr_db: 0.0, count: 0 }
  }
  let mut signal_energy = 0.0
  let mut error_energy = 0.0
  let mut max_abs = 0.0
  let mut sum_abs = 0.0
  for i = 0; i < n; i = i + 1 {
    let x = values[i]
    let back = dequantize(quantize(x, params, mode), params)
    let err = back - x
    let abs_err = if err < 0.0 { -err } else { err }
    signal_energy = signal_energy + x * x
    error_energy = error_energy + err * err
    sum_abs = sum_abs + abs_err
    if abs_err > max_abs {
      max_abs = abs_err
    }
  }
  let snr = if error_energy == 0.0 {
    double_infinity()
  } else if signal_energy == 0.0 {
    0.0
  } else {
    10.0 * @math.log10(signal_energy / error_energy)
  }
  {
    max_abs_error: max_abs,
    mean_abs_error: sum_abs / Double::from_int(n),
    snr_db: snr,
    count: n,
  }
}
