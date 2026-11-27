/// Minimal reader for GGUF containers.
///
/// The point is the type layer, so the parser only keeps what a numeric
/// consumer needs: the tensor table, its ggml types and the location of the
/// payload. Metadata values are skipped rather than interpreted, except for
/// `general.alignment`, which decides where the payload starts.
///
/// Supported tensor types: 0 (`F32`) and 2 (`Q4_0`). Anything else is
/// reported with a byte size of zero instead of being guessed at.

/// The 24-byte fixed part of a GGUF file: magic, version, and the two counts.
/// Enough to identify a file and size it before deciding to parse the rest,
/// which matters when the payload is hundreds of megabytes.
pub struct GgufHeader {
  version : Int
  tensor_count : Int
  kv_count : Int
}

/// Reads only the fixed header. Unlike `parse_gguf` this accepts a buffer
/// that holds just the first 24 bytes, so a caller can identify a file
/// without holding it in memory.
pub fn parse_gguf_header(bytes : Array[Byte]) -> Option[GgufHeader] {
  if bytes.length() < 24 {
    return None
  }
  if byte_at(bytes, 0) != 71 ||
    byte_at(bytes, 1) != 71 ||
    byte_at(bytes, 2) != 85 ||
    byte_at(bytes, 3) != 70 {
    return None
  }
  let version = read_u32(bytes, 4)
  if version < 1 || version > 3 {
    return None
  }
  Some({
    version,
    tensor_count: read_u64(bytes, 8).to_int(),
    kv_count: read_u64(bytes, 16).to_int(),
  })
}

/// One tensor entry from the header.
pub struct GgufTensor {
  name : String
  dims : Array[Int]
  /// ggml type id. 0 = F32, 2 = Q4_0.
  kind : Int
  /// Byte offset from the start of the payload section.
  offset : Int
}

/// A parsed header. The payload bytes stay in the caller's array.
pub struct GgufFile {
  version : Int
  alignment : Int
  tensors : Array[GgufTensor]
  /// Index of the first payload byte.
  data_start : Int
}

fn byte_at(b : Array[Byte], o : Int) -> Int {
  b[o].to_int()
}

fn read_u16(b : Array[Byte], o : Int) -> Int {
  byte_at(b, o) | (byte_at(b, o + 1) << 8)
}

/// Reads four bytes little-endian. The result may be negative: bit 31 is
/// the sign bit of a 32-bit word here, which is exactly what a float
/// reinterpretation wants.
fn read_u32(b : Array[Byte], o : Int) -> Int {
  byte_at(b, o) | (byte_at(b, o + 1) << 8) | (byte_at(b, o + 2) << 16) |
  (byte_at(b, o + 3) << 24)
}

fn read_u64(b : Array[Byte], o : Int) -> Int64 {
  let mut value = Int64::from_int(0)
  for i = 0; i < 8; i = i + 1 {
    let part = Int64::from_int(byte_at(b, o + i))
    value = value | (part << (8 * i))
  }
  value
}

fn ascii_string(b : Array[Byte], start : Int, len : Int) -> String {
  let mut out = ""
  for i = 0; i < len; i = i + 1 {
    let code = byte_at(b, start + i)
    out = out + code.unsafe_to_char().to_string()
  }
  out
}

/// Reads a length-prefixed utf-8 string, returning the text and the offset
/// just past it.
fn read_string(b : Array[Byte], o : Int) -> (String, Int) {
  let len = read_u64(b, o).to_int()
  (ascii_string(b, o + 8, len), o + 8 + len)
}

/// Advances past a metadata value of the given type, returning the new
/// offset. Arrays recurse.
fn skip_value(b : Array[Byte], o : Int, vtype : Int) -> Int {
  if vtype == 0 || vtype == 1 || vtype == 7 {
    o + 1
  } else if vtype == 2 || vtype == 3 {
    o + 2
  } else if vtype == 4 || vtype == 5 || vtype == 6 {
    o + 4
  } else if vtype == 10 || vtype == 11 || vtype == 12 {
    o + 8
  } else if vtype == 8 {
    let (_, next) = read_string(b, o)
    next
  } else if vtype == 9 {
    let elem = read_u32(b, o)
    let count = read_u64(b, o + 4).to_int()
    let mut p = o + 12
    for i = 0; i < count; i = i + 1 {
      p = skip_value(b, p, elem)
    }
    p
  } else {
    // Unknown type: stop rather than read garbage.
    o
  }
}

fn gguf_hex_digit(code : Int) -> Int {
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

/// Decode a hexadecimal string into bytes. Exists because the executable
/// and the tests both need to carry a container around as text: there is
/// no file access on the targets this project builds.
pub fn bytes_from_hex(s : String) -> Array[Byte] {
  let out : Array[Byte] = []
  let mut i = 0
  while i + 1 < s.length() {
    let value = (gguf_hex_digit(s[i].to_int()) << 4) | gguf_hex_digit(s[i + 1].to_int())
    out.push(value.to_byte())
    i = i + 2
  }
  out
}

/// Parses a GGUF header. Returns `None` when the magic or a length check
/// fails, never a half-built value.
pub fn parse_gguf(bytes : Array[Byte]) -> Option[GgufFile] {
  let n = bytes.length()
  if n < 24 {
    return None
  }
  if byte_at(bytes, 0) != 71 ||
    byte_at(bytes, 1) != 71 ||
    byte_at(bytes, 2) != 85 ||
    byte_at(bytes, 3) != 70 {
    return None
  }
  let version = read_u32(bytes, 4)
  if version != 3 {
    return None
  }
  let tensor_count = read_u64(bytes, 8).to_int()
  let kv_count = read_u64(bytes, 16).to_int()
  let mut o = 24
  let mut alignment = 32
  for i = 0; i < kv_count; i = i + 1 {
    if o >= n {
      return None
    }
    let (key, next) = read_string(bytes, o)
    o = next
    let vtype = read_u32(bytes, o)
    o = o + 4
    if key == "general.alignment" && vtype == 4 {
      alignment = read_u32(bytes, o)
    }
    o = skip_value(bytes, o, vtype)
  }
  let tensors : Array[GgufTensor] = []
  for i = 0; i < tensor_count; i = i + 1 {
    if o >= n {
      return None
    }
    let (name, next) = read_string(bytes, o)
    o = next
    let n_dims = read_u32(bytes, o)
    o = o + 4
    let dims : Array[Int] = Array::make(n_dims, 0)
    for d = 0; d < n_dims; d = d + 1 {
      dims[d] = read_u64(bytes, o).to_int()
      o = o + 8
    }
    let kind = read_u32(bytes, o)
    o = o + 4
    let offset = read_u64(bytes, o).to_int()
    o = o + 8
    tensors.push({ name, dims, kind, offset })
  }
  let mut data_start = o
  while data_start % alignment != 0 {
    data_start = data_start + 1
  }
  // A header can be well formed and still describe payload that is not
  // there. Check every advertised extent against the buffer instead of
  // letting a later read go past the end.
  if data_start > n {
    return None
  }
  let payload_len = n - data_start
  for i = 0; i < tensors.length(); i = i + 1 {
    let t = tensors[i]
    let size = t.byte_size()
    if t.offset < 0 || (size > 0 && t.offset + size > payload_len) {
      return None
    }
  }
  Some({ version, alignment, tensors, data_start })
}

/// Number of scalar elements in the tensor.
pub fn GgufTensor::element_count(self : GgufTensor) -> Int {
  let mut count = 1
  for i = 0; i < self.dims.length(); i = i + 1 {
    count = count * self.dims[i]
  }
  count
}

/// Bytes occupied by the tensor payload. Zero for unsupported types.
pub fn GgufTensor::byte_size(self : GgufTensor) -> Int {
  let n = self.element_count()
  if self.kind == 0 {
    n * 4
  } else if self.kind == 2 {
    (n / 32) * 18
  } else if self.kind == 6 {
    (n / 32) * 22
  } else if self.kind == 8 {
    (n / 32) * 34
  } else if self.kind == 10 {
    (n / 256) * 84
  } else if self.kind == 11 {
    (n / 256) * 110
  } else if self.kind == 12 {
    (n / 256) * 144
  } else if self.kind == 13 {
    (n / 256) * 176
  } else if self.kind == 14 {
    (n / 256) * 210
  } else {
    0
  }
}

/// Dequantizes a tensor into binary64 values. F32 is widened directly; the
/// block formats go block by block, 32 weights at a time: Q4_0 in 18 bytes,
/// Q5_0 in 22 (two bytes of scale, four of high bits, sixteen packed nibbles),
/// Q8_0 in 34 (two bytes of scale, thirty-two signed bytes), Q4_K in 144,
/// Q2_K in 84, Q3_K in 110 and Q6_K in 210 for 256 weights (the K-quant
/// super-block used by the Q2_K, Q3_K_M, Q3_K_L, Q4_K_M and Q6_K files
/// llama.cpp ships).
pub fn dequantize_tensor(
  bytes : Array[Byte],
  file : GgufFile,
  t : GgufTensor,
) -> Array[Double] {
  let n = t.element_count()
  let base = file.data_start + t.offset
  let out = Array::make(n, 0.0)
  if t.kind == 0 {
    for i = 0; i < n; i = i + 1 {
      out[i] = Float::reinterpret_from_int(read_u32(bytes, base + i * 4)).to_double()
    }
  } else if t.kind == 2 {
    let blocks = n / 32
    for b = 0; b < blocks; b = b + 1 {
      let o = base + b * 18
      let scale = F16::from_bits(read_u16(bytes, o)).to_double()
      for j = 0; j < 16; j = j + 1 {
        let packed = byte_at(bytes, o + 2 + j)
        let low = (packed & 15) - 8
        let high = ((packed >> 4) & 15) - 8
        out[b * 32 + j] = Double::from_int(low) * scale
        out[b * 32 + j + 16] = Double::from_int(high) * scale
      }
    }
  } else if t.kind == 6 {
    // Q5_0: the fifth bit of each weight lives in the four qh bytes, weight j
    // taking bit j and weight j+16 taking bit j+16.
    let blocks = n / 32
    for b = 0; b < blocks; b = b + 1 {
      let o = base + b * 22
      let scale = F16::from_bits(read_u16(bytes, o)).to_double()
      let qh = read_u32(bytes, o + 2)
      for j = 0; j < 16; j = j + 1 {
        let packed = byte_at(bytes, o + 6 + j)
        let low5 = ((packed & 15) | (((qh >> j) & 1) << 4)) - 16
        let high5 = ((packed >> 4) | (((qh >> (j + 16)) & 1) << 4)) - 16
        out[b * 32 + j] = Double::from_int(low5) * scale
        out[b * 32 + j + 16] = Double::from_int(high5) * scale
      }
    }
  } else if t.kind == 8 {
    let blocks = n / 32
    for b = 0; b < blocks; b = b + 1 {
      let o = base + b * 34
      let scale = F16::from_bits(read_u16(bytes, o)).to_double()
      for j = 0; j < 32; j = j + 1 {
        let raw = byte_at(bytes, o + 2 + j)
        let signed = if raw >= 128 { raw - 256 } else { raw }
        out[b * 32 + j] = Double::from_int(signed) * scale
      }
    }
  } else if t.kind == 10 {
    // Q2_K: 256 weights in 84 bytes. Sixteen one-byte scales split into a low
    // nibble that multiplies `d` and a high nibble that multiplies `dmin`,
    // then 64 bytes of four 2-bit weights each, then f16 `d` and f16 `dmin`.
    // A 32-byte run covers 128 weights: four bit pairs, each giving sixteen
    // values from the low half and sixteen from the high half.
    let blocks = n / 256
    for b = 0; b < blocks; b = b + 1 {
      let o = base + b * 84
      let d = F16::from_bits(read_u16(bytes, o + 80)).to_double()
      let dmin = F16::from_bits(read_u16(bytes, o + 82)).to_double()
      for seg = 0; seg < 2; seg = seg + 1 {
        let q = o + 16 + seg * 32
        for j = 0; j < 4; j = j + 1 {
          let shift = j * 2
          let s1 = byte_at(bytes, o + seg * 8 + j * 2)
          let s2 = byte_at(bytes, o + seg * 8 + j * 2 + 1)
          let dl1 = d * Double::from_int(s1 & 15)
          let ml1 = dmin * Double::from_int(s1 >> 4)
          let dl2 = d * Double::from_int(s2 & 15)
          let ml2 = dmin * Double::from_int(s2 >> 4)
          let base_out = b * 256 + seg * 128 + j * 32
          for l = 0; l < 16; l = l + 1 {
            out[base_out + l] = dl1 * Double::from_int((byte_at(bytes, q + l) >> shift) & 3) - ml1
            out[base_out + 16 + l] = dl2 * Double::from_int((byte_at(bytes, q + 16 + l) >> shift) & 3) - ml2
          }
        }
      }
    }
  } else if t.kind == 11 {
    // Q3_K: 256 weights in 110 bytes: 32 high-bit bytes, 64 bytes of 2-bit
    // weights, twelve bytes carrying sixteen 6-bit scales, f16 `d`. A weight
    // is three bits, and the high bit inverts the offset: it is subtracted
    // four when the mask bit is clear, which is where llama.cpp flips the
    // bit and shifts the mask instead of moving the pointer.
    let blocks = n / 256
    for b = 0; b < blocks; b = b + 1 {
      let o = base + b * 110
      let d = F16::from_bits(read_u16(bytes, o + 108)).to_double()
      for seg = 0; seg < 2; seg = seg + 1 {
        let qs = o + 32 + seg * 32
        let hm = o
        for j = 0; j < 4; j = j + 1 {
          let shift = j * 2
          let bit = seg * 4 + j
          let dl1 = d * Double::from_int(k3_scale(bytes, o + 96, seg * 8 + j * 2))
          let dl2 = d * Double::from_int(k3_scale(bytes, o + 96, seg * 8 + j * 2 + 1))
          let base_out = b * 256 + seg * 128 + j * 32
          for l = 0; l < 16; l = l + 1 {
            let off1 = if ((byte_at(bytes, hm + l) >> bit) & 1) == 1 { 0 } else { 4 }
            let off2 = if ((byte_at(bytes, hm + 16 + l) >> bit) & 1) == 1 { 0 } else { 4 }
            out[base_out + l] = dl1 * Double::from_int(((byte_at(bytes, qs + l) >> shift) & 3) - off1)
            out[base_out + 16 + l] = dl2 * Double::from_int(((byte_at(bytes, qs + 16 + l) >> shift) & 3) - off2)
          }
        }
      }
    }
  } else if t.kind == 12 {
    // Q4_K: 256 weights in 144 bytes. One f16 scale `d` multiplies the 6-bit
    // sub-scales, one f16 `dmin` multiplies the 6-bit mins, and the weights
    // are 4-bit unsigned with `d * sc * q - dmin * m` applied per 32 weights.
    // The twenty-four packed scale bits are unpacked exactly as llama.cpp's
    // get_scale_min_k4 does.
    let blocks = n / 256
    for b = 0; b < blocks; b = b + 1 {
      let o = base + b * 144
      let d = F16::from_bits(read_u16(bytes, o)).to_double()
      let dmin = F16::from_bits(read_u16(bytes, o + 2)).to_double()
      for g = 0; g < 4; g = g + 1 {
        let si = g * 2
        let (sc1, m1) = k4_scale_min(bytes, o + 4, si)
        let (sc2, m2) = k4_scale_min(bytes, o + 4, si + 1)
        let q = o + 16 + g * 32
        let ds1 = d * Double::from_int(sc1)
        let dm1 = dmin * Double::from_int(m1)
        let ds2 = d * Double::from_int(sc2)
        let dm2 = dmin * Double::from_int(m2)
        for l = 0; l < 32; l = l + 1 {
          let packed = byte_at(bytes, q + l)
          out[b * 256 + g * 64 + l] = ds1 * Double::from_int(packed & 15) - dm1
          out[b * 256 + g * 64 + 32 + l] = ds2 * Double::from_int(packed >> 4) - dm2
        }
      }
    }
  } else if t.kind == 13 {
    // Q5_K: 256 weights in 176 bytes. Same twelve packed scale/min bytes as
    // Q4_K, but the fifth bit of every weight lives in 32 shared bytes: for
    // group g the low nibbles take bit 2g and the high nibbles bit 2g+1 of
    // the same byte, which is why llama.cpp shifts a mask instead of moving
    // the pointer. The weight is unsigned and offset by `dmin * m`.
    let blocks = n / 256
    for b = 0; b < blocks; b = b + 1 {
      let o = base + b * 176
      let d = F16::from_bits(read_u16(bytes, o)).to_double()
      let dmin = F16::from_bits(read_u16(bytes, o + 2)).to_double()
      let qh = o + 16
      let ql = o + 48
      for g = 0; g < 4; g = g + 1 {
        let si = g * 2
        let (sc1, m1) = k4_scale_min(bytes, o + 4, si)
        let (sc2, m2) = k4_scale_min(bytes, o + 4, si + 1)
        let ds1 = d * Double::from_int(sc1)
        let dm1 = dmin * Double::from_int(m1)
        let ds2 = d * Double::from_int(sc2)
        let dm2 = dmin * Double::from_int(m2)
        for l = 0; l < 32; l = l + 1 {
          let packed = byte_at(bytes, ql + g * 32 + l)
          let high = byte_at(bytes, qh + l)
          let fifth_low = if ((high >> (2 * g)) & 1) == 1 { 16 } else { 0 }
          let fifth_high = if ((high >> (2 * g + 1)) & 1) == 1 { 16 } else { 0 }
          out[b * 256 + g * 64 + l] =
            ds1 * Double::from_int((packed & 15) + fifth_low) - dm1
          out[b * 256 + g * 64 + 32 + l] =
            ds2 * Double::from_int((packed >> 4) + fifth_high) - dm2
        }
      }
    }
  } else if t.kind == 14 {
    // Q6_K: 256 weights in 210 bytes: 128 low nibbles, 64 high-bit pairs,
    // 16 signed sub-scales and one f16 scale. Weights are 6-bit signed
    // (biased by 32), scales are int8.
    let blocks = n / 256
    for b = 0; b < blocks; b = b + 1 {
      let o = base + b * 210
      let d = F16::from_bits(read_u16(bytes, o + 208)).to_double()
      for seg = 0; seg < 2; seg = seg + 1 {
        let ql = o + seg * 64
        let qh = o + 128 + seg * 32
        let sc = o + 192 + seg * 8
        for l = 0; l < 32; l = l + 1 {
          let si = l / 16
          let h = byte_at(bytes, qh + l)
          let low = byte_at(bytes, ql + l)
          let high = byte_at(bytes, ql + l + 32)
          let q1 = ((low & 15) | ((h & 3) << 4)) - 32
          let q2 = ((high & 15) | (((h >> 2) & 3) << 4)) - 32
          let q3 = ((low >> 4) | (((h >> 4) & 3) << 4)) - 32
          let q4 = ((high >> 4) | (((h >> 6) & 3) << 4)) - 32
          let base_out = b * 256 + seg * 128
          out[base_out + l] = d * signed_byte(bytes, sc + si) * Double::from_int(q1)
          out[base_out + l + 32] = d * signed_byte(bytes, sc + si + 2) * Double::from_int(q2)
          out[base_out + l + 64] = d * signed_byte(bytes, sc + si + 4) * Double::from_int(q3)
          out[base_out + l + 96] = d * signed_byte(bytes, sc + si + 6) * Double::from_int(q4)
        }
      }
    }
  }
  out
}

/// Q3_K's sixteen 6-bit scales. Below eight, the low four bits are the low
/// nibble of byte `i`; from eight on they are the high nibble of byte `i - 8`.
/// The two high bits sit in the second four bytes, one field of two bits per
/// byte and per group of four scales.
fn k3_scale(b : Array[Byte], sbase : Int, i : Int) -> Int {
  let low = if i < 8 {
    byte_at(b, sbase + i) & 15
  } else {
    byte_at(b, sbase + i - 8) >> 4
  }
  let high = (byte_at(b, sbase + 8 + (i % 4)) >> (2 * (i / 4))) & 3
  (low | (high << 4)) - 32
}

/// llama.cpp's get_scale_min_k4: the first four sub-scales and mins sit in the
/// low six bits of `q[0..4]` and `q[4..8]`; the high two bits are stored in
/// `q[0..4]` (scales) and `q[4..8]` (mins) again, six at a time, for the
/// second four, whose low four bits live in the nibbles of `q[8..12]`.
fn k4_scale_min(b : Array[Byte], qbase : Int, j : Int) -> (Int, Int) {
  if j < 4 {
    (byte_at(b, qbase + j) & 63, byte_at(b, qbase + j + 4) & 63)
  } else {
    let sc = (byte_at(b, qbase + j + 4) & 15) | ((byte_at(b, qbase + j - 4) >> 6) << 4)
    let mn = (byte_at(b, qbase + j + 4) >> 4) | ((byte_at(b, qbase + j) >> 6) << 4)
    (sc, mn)
  }
}

/// A byte read as a signed 8-bit integer.
fn signed_byte(b : Array[Byte], o : Int) -> Double {
  let raw = byte_at(b, o)
  Double::from_int(if raw >= 128 { raw - 256 } else { raw })
}
