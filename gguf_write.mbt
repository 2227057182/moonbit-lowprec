/// Writing GGUF containers, and the two quantizers that make a payload worth
/// writing: `Q8_0` and `Q4_0`. Together with the reader they close the loop —
/// bytes a caller builds here parse back with the same tensor table, and a
/// file written here is readable by llama.cpp's own tools.

/// One tensor to write: its name, shape, ggml type and payload bytes. The
/// payload must already match the type, since the writer does not quantize
/// on its own; `quantize_q8_0` and `quantize_q4_0` produce the two block
/// formats.
pub(all) struct GgufTensorOut {
  name : String
  dims : Array[Int]
  kind : Int
  payload : Array[Byte]
}

fn align_up(x : Int, a : Int) -> Int {
  let rem = x % a
  if rem == 0 {
    x
  } else {
    x + (a - rem)
  }
}

fn put_u32(out : Array[Byte], at : Int, v : Int) -> Unit {
  out[at] = Int::to_byte(v & 255)
  out[at + 1] = Int::to_byte((v >> 8) & 255)
  out[at + 2] = Int::to_byte((v >> 16) & 255)
  out[at + 3] = Int::to_byte((v >> 24) & 255)
}

fn put_u64(out : Array[Byte], at : Int, v : Int) -> Unit {
  put_u32(out, at, v)
  put_u32(out, at + 4, 0)
}

fn put_bytes(out : Array[Byte], at : Int, src : Array[Byte]) -> Unit {
  for i = 0; i < src.length(); i = i + 1 {
    out[at + i] = src[i]
  }
}

/// Builds a version-3 container with no metadata. Tensors are placed in the
/// order given, each payload starting at the next multiple of `alignment`, and
/// their offsets are recorded relative to the payload section, as the format
/// specifies. Metadata is omitted deliberately: `kv_count` zero is a valid
/// file and keeps the writer to what a numeric caller needs.
pub fn build_gguf(tensors : Array[GgufTensorOut], alignment : Int) -> Array[Byte] {
  let mut header = 24
  for t in tensors {
    header = header + 8 + t.name.length() + 4 + 8 * t.dims.length() + 4 + 8
  }
  let data_start = align_up(header, alignment)
  let offsets : Array[Int] = Array::make(tensors.length(), 0)
  let mut total = data_start
  for i = 0; i < tensors.length(); i = i + 1 {
    total = align_up(total, alignment)
    offsets[i] = total - data_start
    total = total + tensors[i].payload.length()
  }
  let out : Array[Byte] = Array::make(total, Int::to_byte(0))
  out[0] = Int::to_byte(71)
  out[1] = Int::to_byte(71)
  out[2] = Int::to_byte(85)
  out[3] = Int::to_byte(70)
  put_u32(out, 4, 3)
  put_u64(out, 8, tensors.length())
  put_u64(out, 16, 0)
  let mut o = 24
  for i = 0; i < tensors.length(); i = i + 1 {
    let t = tensors[i]
    put_u64(out, o, t.name.length())
    for c = 0; c < t.name.length(); c = c + 1 {
      out[o + 8 + c] = Int::to_byte(t.name[c].to_int())
    }
    o = o + 8 + t.name.length()
    put_u32(out, o, t.dims.length())
    o = o + 4
    for d = 0; d < t.dims.length(); d = d + 1 {
      put_u64(out, o, t.dims[d])
      o = o + 8
    }
    put_u32(out, o, t.kind)
    o = o + 4
    put_u64(out, o, offsets[i])
    o = o + 8
  }
  for i = 0; i < tensors.length(); i = i + 1 {
    put_bytes(out, data_start + offsets[i], tensors[i].payload)
  }
  out
}

fn round_half_away(x : Double) -> Int {
  if x >= 0.0 {
    (x + 0.5).to_int()
  } else {
    -((-x + 0.5).to_int())
  }
}

fn clamp_int(x : Int, lo : Int, hi : Int) -> Int {
  if x < lo {
    lo
  } else if x > hi {
    hi
  } else {
    x
  }
}

fn code_byte(q : Int) -> Int {
  if q < 0 {
    q + 256
  } else {
    q
  }
}

/// Quantizes doubles into thirty-two-weight `Q8_0` blocks: one f16 scale per
/// block at `max / 127`, then 32 signed bytes, each the rounded quotient.
pub fn quantize_q8_0(values : Array[Double]) -> Array[Byte] {
  let blocks = values.length() / 32
  let out : Array[Byte] = Array::make(blocks * 34, Int::to_byte(0))
  for b = 0; b < blocks; b = b + 1 {
    let mut amax = 0.0
    for j = 0; j < 32; j = j + 1 {
      let v = values[b * 32 + j]
      let a = if v < 0.0 { -v } else { v }
      if a > amax {
        amax = a
      }
    }
    let scale = F16::from_float(Float::from_double(amax / 127.0))
    let d = scale.to_double()
    let o = b * 34
    out[o] = Int::to_byte(scale.bits() & 255)
    out[o + 1] = Int::to_byte((scale.bits() >> 8) & 255)
    for j = 0; j < 32; j = j + 1 {
      let q = if d == 0.0 {
        0
      } else {
        clamp_int(round_half_away(values[b * 32 + j] / d), -127, 127)
      }
      out[o + 2 + j] = Int::to_byte(code_byte(q) & 255)
    }
  }
  out
}

/// Quantizes doubles into thirty-two-weight `Q4_0` blocks: one f16 scale per
/// block at `max / 8`, then sixteen bytes. Weight j is the low nibble and
/// weight j + 16 the high one, both offset by eight so the range is -8..7.
pub fn quantize_q4_0(values : Array[Double]) -> Array[Byte] {
  let blocks = values.length() / 32
  let out : Array[Byte] = Array::make(blocks * 18, Int::to_byte(0))
  for b = 0; b < blocks; b = b + 1 {
    let mut amax = 0.0
    for j = 0; j < 32; j = j + 1 {
      let v = values[b * 32 + j]
      let a = if v < 0.0 { -v } else { v }
      if a > amax {
        amax = a
      }
    }
    let scale = F16::from_float(Float::from_double(amax / 8.0))
    let d = scale.to_double()
    let o = b * 18
    out[o] = Int::to_byte(scale.bits() & 255)
    out[o + 1] = Int::to_byte((scale.bits() >> 8) & 255)
    for j = 0; j < 16; j = j + 1 {
      let q1 = if d == 0.0 {
        8
      } else {
        round_half_away(values[b * 32 + j] / d) + 8
      }
      let q2 = if d == 0.0 {
        8
      } else {
        round_half_away(values[b * 32 + 16 + j] / d) + 8
      }
      let lo = clamp_int(q1, 0, 15)
      let hi = clamp_int(q2, 0, 15)
      out[o + 2 + j] = Int::to_byte(lo | (hi << 4))
    }
  }
  out
}
