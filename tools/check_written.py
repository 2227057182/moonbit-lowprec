"""Reads a container written by this project and checks it with the official
`gguf` package, the implementation maintained by the organisation that
publishes llama.cpp.

The container comes from `moon run cmd/main -- emit-gguf`, which prints
`bytes=N hex=<...>`. Passing the hex file here means the bytes are read by a
reader this repository did not write.

    python3 tools/check_written.py written.hex
"""

import sys

import numpy as np

import gguf


def main():
    hexs = open(sys.argv[1], encoding="utf-8").read().strip()
    # The inspector prints `bytes=210 hex=<...>`; take whatever follows `hex=`.
    if "hex=" in hexs:
        hexs = hexs.split("hex=")[-1].strip()
    body = bytes.fromhex(hexs)
    path = sys.argv[1].replace(".hex", ".gguf")
    with open(path, "wb") as f:
        f.write(body)

    reader = gguf.GGUFReader(path)
    values = np.array([(i - 16) / 16.0 for i in range(32)], dtype=np.float64)
    errors = 0
    print(f"{len(body)} bytes, magic {body[:4]!r}")
    for t in reader.tensors:
        q = np.asarray(
            gguf.quants.dequantize(np.frombuffer(t.data.tobytes(), dtype=np.uint8), t.tensor_type)
        )
        err = float(np.abs(q - values).max())
        print(f"  {t.name} type {int(t.tensor_type)} shape {[int(d) for d in t.shape]} maxerr {err}")
        if t.name == "block_q8_0" and not err <= 0.004:
            errors += 1
        if t.name == "block_q4_0" and not err <= 0.0625:
            errors += 1
    if len(reader.tensors) != 2:
        errors += 1
    print("written container: read back with errors" if errors else
          "written container: read back by the official package, 0 errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
