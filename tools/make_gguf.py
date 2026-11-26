"""Build the 324-byte GGUF v3 container used by the tests.

Layout follows the published format:
  header (magic, version, tensor_count, metadata_kv_count)
  metadata kv pairs
  tensor infos
  padding to alignment
  tensor data

Run: python tools/make_gguf.py
it writes tiny.gguf (324 bytes) and prints the same hex the tests embed.
"""

import struct

ALIGN = 32


def u32(v):
    return struct.pack('<I', v)


def u64(v):
    return struct.pack('<Q', v)


def s(text):
    raw = text.encode('utf-8')
    return u64(len(raw)) + raw


header = b'GGUF' + u32(3)

meta = []
meta.append(s('general.architecture') + u32(8) + s('llama'))
meta.append(s('general.alignment') + u32(4) + u32(ALIGN))
meta.append(s('test.count') + u32(5) + struct.pack('<i', 3))

# Tensor 0: Q4_0, 64 weights in two blocks.
# Block 0: scale 0.5, sixteen zero bytes -> every weight -4.0.
# Block 1: scale 0.5, sixteen bytes of 0x1F. The low nibble is 15 -> +3.5
# for weight j, the high nibble is 1 -> -3.5 for weight j+16.
q4_block0 = struct.pack('<H', 0x3800) + bytes(16)
q4_block1 = struct.pack('<H', 0x3800) + bytes([0x1F] * 16)
q4_data = q4_block0 + q4_block1
q4_size = len(q4_data)

# Tensor 1: F32, 8 weights.
f32_values = [0.5, -0.25, 1.0, -1.0, 2.5, 0.0, -0.75, 0.125]
f32_data = b''.join(struct.pack('<f', v) for v in f32_values)

tensors = []
tensors.append(s('token_embd.weight') + u32(2) + u64(32) + u64(2) + u32(2) + u64(0))
tensors.append(s('output_norm.weight') + u32(1) + u64(8) + u32(0) + u64(q4_size))

body = header + u64(len(tensors)) + u64(len(meta))
body += b''.join(meta)
body += b''.join(tensors)

while len(body) % ALIGN != 0:
    body += b'\x00'

body += q4_data + f32_data

with open('tiny.gguf', 'wb') as handle:
    handle.write(body)

print('size', len(body))
print('hex', body.hex())
