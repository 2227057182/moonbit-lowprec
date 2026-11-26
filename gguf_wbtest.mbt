/// A 324-byte GGUF v3 container written by an independent Python script:
/// two metadata pairs, one Q4_0 tensor (two blocks, 64 weights) and one
/// F32 tensor of 8 values. The payload is synthetic, the layout is the
/// published format, so this exercises the reader end to end without
/// needing a model download.
let tiny_gguf_hex = "474755460300000002000000000000000300000000000000140000000000000067656e6572616c2e6172636869746563747572650800000005000000000000006c6c616d61110000000000000067656e6572616c2e616c69676e6d656e7404000000200000000a00000000000000746573742e636f756e7405000000030000001100000000000000746f6b656e5f656d62642e776569676874020000002000000000000000020000000000000002000000000000000000000012000000000000006f75747075745f6e6f726d2e77656967687401000000080000000000000000000000240000000000000000000000000000000000000000000000000000000000380000000000000000000000000000000000381f1f1f1f1f1f1f1f1f1f1f1f1f1f1f1f0000003f000080be0000803f000080bf0000204000000000000040bf0000003e"

fn hex_to_bytes(s : String) -> Array[Byte] {
  bytes_from_hex(s)
}

test "parse the tiny gguf header" {
  let bytes = hex_to_bytes(tiny_gguf_hex)
  assert_eq(bytes.length(), 324)
  match parse_gguf(bytes) {
    Some(f) => {
      assert_eq(f.version, 3)
      assert_eq(f.alignment, 32)
      assert_eq(f.tensors.length(), 2)
      assert_eq(f.tensors[0].name, "token_embd.weight")
      assert_eq(f.tensors[0].dims.length(), 2)
      assert_eq(f.tensors[0].dims[0], 32)
      assert_eq(f.tensors[0].dims[1], 2)
      assert_eq(f.tensors[0].kind, 2)
      assert_eq(f.tensors[0].offset, 0)
      assert_eq(f.tensors[0].element_count(), 64)
      assert_eq(f.tensors[0].byte_size(), 36)
      assert_eq(f.tensors[1].name, "output_norm.weight")
      assert_eq(f.tensors[1].dims.length(), 1)
      assert_eq(f.tensors[1].dims[0], 8)
      assert_eq(f.tensors[1].kind, 0)
      assert_eq(f.tensors[1].byte_size(), 32)
    }
    None => assert_eq("parsed", "None")
  }
}

test "dequantize a q4_0 tensor and an f32 tensor" {
  let bytes = hex_to_bytes(tiny_gguf_hex)
  match parse_gguf(bytes) {
    Some(f) => {
      let q4 = dequantize_tensor(bytes, f, f.tensors[0])
      assert_eq(q4.length(), 64)
      for i = 0; i < 32; i = i + 1 {
        assert_eq(q4[i], -4.0)
      }
      // Byte 0x1F: low nibble 15 -> +7 steps, high nibble 1 -> -7 steps.
      // GGML stores weight j in the low nibble and weight j+16 in the high
      // nibble of the same byte.
      for i = 0; i < 16; i = i + 1 {
        assert_eq(q4[32 + i], 3.5)
      }
      for i = 0; i < 16; i = i + 1 {
        assert_eq(q4[48 + i], -3.5)
      }
      let f32 = dequantize_tensor(bytes, f, f.tensors[1])
      assert_eq(f32.length(), 8)
      assert_eq(f32[0], 0.5)
      assert_eq(f32[1], -0.25)
      assert_eq(f32[2], 1.0)
      assert_eq(f32[3], -1.0)
      assert_eq(f32[4], 2.5)
      assert_eq(f32[5], 0.0)
      assert_eq(f32[6], -0.75)
      assert_eq(f32[7], 0.125)
    }
    None => assert_eq("parsed", "None")
  }
}

test "a non-gguf byte array is rejected" {
  let junk : Array[Byte] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
  match parse_gguf(junk) {
    Some(_) => assert_eq("rejected", "accepted")
    None => assert_eq(true, true)
  }
}
