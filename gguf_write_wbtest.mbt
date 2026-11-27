fn slice_bytes(b : Array[Byte], start : Int, len : Int) -> Array[Byte] {
  let out : Array[Byte] = Array::make(len, Int::to_byte(0))
  for i = 0; i < len; i = i + 1 {
    out[i] = b[start + i]
  }
  out
}

fn same_bytes(a : Array[Byte], b : Array[Byte]) -> Bool {
  if a.length() != b.length() {
    return false
  }
  let mut ok = true
  for i = 0; i < a.length(); i = i + 1 {
    if a[i].to_int() != b[i].to_int() {
      ok = false
    }
  }
  ok
}

test "a container written here reads back with the same payloads" {
  // Every block of the fixture is copied out, written into a fresh container
  // with no metadata, and read back: names, kinds, sizes and payload bytes
  // must be unchanged, which is the round trip the writer is for.
  let bytes = bytes_from_hex(fixture_gguf_hex)
  let file = parse_gguf(bytes).unwrap()
  let outs : Array[GgufTensorOut] = []
  for t in file.tensors {
    outs.push({
      name: t.name,
      dims: t.dims,
      kind: t.kind,
      payload: slice_bytes(bytes, file.data_start + t.offset, t.byte_size()),
    })
  }
  let written = build_gguf(outs, 32)
  let back = parse_gguf(written).unwrap()
  assert_eq(back.version, 3)
  assert_eq(back.alignment, 32)
  assert_eq(back.tensors.length(), 9)
  for i = 0; i < 9; i = i + 1 {
    assert_eq(back.tensors[i].name, file.tensors[i].name)
    assert_eq(back.tensors[i].kind, file.tensors[i].kind)
    assert_eq(back.tensors[i].byte_size(), file.tensors[i].byte_size())
    let a = slice_bytes(
      written,
      back.data_start + back.tensors[i].offset,
      back.tensors[i].byte_size(),
    )
    let b = slice_bytes(
      bytes,
      file.data_start + file.tensors[i].offset,
      file.tensors[i].byte_size(),
    )
    assert_eq(same_bytes(a, b), true)
  }
}

test "q8_0 quantization keeps every weight within half a step" {
  let values = Array::make(32, 0.0)
  for i = 0; i < 32; i = i + 1 {
    values[i] = Double::from_int(i - 16) / 16.0
  }
  let payload = quantize_q8_0(values)
  assert_eq(payload.length(), 34)
  // amax is 1.0, so the scale is 1/127 in binary16 and the step is 0.00787..
  let one : Array[GgufTensorOut] = [{ name: "q8", dims: [32], kind: 8, payload }]
  let written = build_gguf(one, 32)
  let file = parse_gguf(written).unwrap()
  let out = dequantize_tensor(written, file, file.tensors[0])
  for i = 0; i < 32; i = i + 1 {
    let diff = out[i] - values[i]
    let a = if diff < 0.0 { -diff } else { diff }
    assert_eq(a <= 0.004, true)
  }
}

test "q4_0 quantization keeps every weight within half a step" {
  // amax is 1.0 and the scale 0.125, so codes land in -8..8 and the step is
  // 0.125; half a step is 0.0625.
  let values = Array::make(32, 0.0)
  for i = 0; i < 32; i = i + 1 {
    values[i] = Double::from_int(i - 16) / 16.0
  }
  let payload = quantize_q4_0(values)
  assert_eq(payload.length(), 18)
  assert_eq(payload[0].to_int(), 0x00)
  assert_eq(payload[1].to_int(), 0x30)
  let one : Array[GgufTensorOut] = [{ name: "q4", dims: [32], kind: 2, payload }]
  let written = build_gguf(one, 32)
  let file = parse_gguf(written).unwrap()
  let out = dequantize_tensor(written, file, file.tensors[0])
  for i = 0; i < 32; i = i + 1 {
    let diff = out[i] - values[i]
    let a = if diff < 0.0 { -diff } else { diff }
    assert_eq(a <= 0.0625, true)
  }
}
