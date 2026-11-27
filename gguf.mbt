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
  } else {
    0
  }
}

/// Dequantizes a tensor into binary64 values. F32 is widened directly;
/// Q4_0 goes block by block, 32 weights per 18-byte block.
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
  }
  out
}
