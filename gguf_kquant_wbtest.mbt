/// The K-quant super-block formats pack eight six-bit scales and eight six-bit
/// mins into twelve bytes, and the second four of each take their high two bits
/// from bytes that already carry the first four. That is the part of the layout
/// most likely to be transcribed wrong, so it is checked here against a table
/// written out by hand rather than against the implementation.

test "k4 scale and min unpacking covers both bit paths" {
  // bytes 0..3 hold scales 1..4, bytes 4..7 hold mins 5..8, the top two bits
  // of bytes 0..3 raise scales 4..7 by sixteen, and the nibbles of bytes 8..11
  // are their low four bits and the low four bits of mins 4..7.
  let b = bytes_from_hex("414243440506070811223344")
  let want : Array[(Int, Int)] = [
    (1, 5), (2, 6), (3, 7), (4, 8), (17, 1), (18, 2), (19, 3), (20, 4),
  ]
  for j = 0; j < 8; j = j + 1 {
    let pair = k4_scale_min(b, 0, j)
    assert_eq(pair.0, want[j].0)
    assert_eq(pair.1, want[j].1)
  }
}

test "k4 scale and min reach the six bit maximum" {
  // 0x3F is 63 in the low six bits and 0x00 above, so the second half keeps
  // only its nibble; 0xFF in the nibble bytes gives 15 for both fields.
  let b = bytes_from_hex("3f03030300000000ffffffff")
  let first = k4_scale_min(b, 0, 0)
  assert_eq(first.0, 63)
  assert_eq(first.1, 0)
  let last = k4_scale_min(b, 0, 4)
  assert_eq(last.0, 15)
  assert_eq(last.1, 15)
}

test "K-quant payload sizes follow the 256-weight super-block" {
  let q4k = GgufTensor::{ name: "a", dims: [256], kind: 12, offset: 0 }
  assert_eq(q4k.byte_size(), 144)
  let q4k2 = GgufTensor::{ name: "a", dims: [512], kind: 12, offset: 0 }
  assert_eq(q4k2.byte_size(), 288)
  let q6k = GgufTensor::{ name: "b", dims: [256], kind: 14, offset: 0 }
  assert_eq(q6k.byte_size(), 210)
  let q6k2 = GgufTensor::{ name: "b", dims: [512], kind: 14, offset: 0 }
  assert_eq(q6k2.byte_size(), 420)
  let q5k = GgufTensor::{ name: "c", dims: [256], kind: 13, offset: 0 }
  assert_eq(q5k.byte_size(), 176)
  let q5k2 = GgufTensor::{ name: "c", dims: [512], kind: 13, offset: 0 }
  assert_eq(q5k2.byte_size(), 352)
}
