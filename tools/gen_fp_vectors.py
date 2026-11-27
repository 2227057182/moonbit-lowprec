"""Generates the reference vector table for the 16-bit arithmetic.

Every operand pair is decoded into a Fraction, the operation is carried out
exactly, and the result is rounded once to the target format by the rules in
IEEE 754-2019: nearest, ties to even, gradual underflow, overflow to infinity.
No floating-point type is involved on the reference side, so the table cannot
inherit a double-rounding defect from an emulator.

As an independent third opinion the binary16 pairs are also evaluated with
numpy's float16, the commonly used baseline. Where the two disagree the exact
result decides, and the disagreement is reported: numpy computes binary16
arithmetic through a wider intermediate, so near a rounding boundary it can
be off by one ulp.

    python3 tools/gen_fp_vectors.py > fp_vectors.mbt

The vectors are emitted as MoonBit source so the table is checked by the
normal test run rather than by a manual step.
"""

import random
import struct
from fractions import Fraction

import numpy as np


def decode(bits, eb, mb, bias):
    """Bit pattern to exact Fraction, or None for NaN and infinity."""
    total = 1 + eb + mb
    sign = bits >> (total - 1)
    exp = (bits >> mb) & ((1 << eb) - 1)
    mant = bits & ((1 << mb) - 1)
    if exp == (1 << eb) - 1:
        return None
    if exp == 0:
        value = Fraction(mant, 2 ** (bias + mb - 1))
    else:
        value = Fraction((1 << mb) + mant) * Fraction(2) ** (exp - bias - mb)
    return -value if sign else value


def floor_log2(x):
    k = x.numerator.bit_length() - x.denominator.bit_length()
    while Fraction(2) ** (k + 1) <= x:
        k += 1
    while Fraction(2) ** k > x:
        k -= 1
    return k


def round_nearest_even(m):
    """Rounds a non-negative Fraction to the nearest integer, ties to even."""
    q, r = divmod(m.numerator, m.denominator)
    twice = 2 * r
    if twice > m.denominator or (twice == m.denominator and q % 2 == 1):
        q += 1
    return q


def encode(value, eb, mb, bias):
    """Exact Fraction to bit pattern, one rounding."""
    total = 1 + eb + mb
    sign = 1 if value < 0 else 0
    x = -value if value < 0 else value
    maxexp = (1 << eb) - 1
    inf = (sign << (total - 1)) | (maxexp << mb)
    if x == 0:
        return sign << (total - 1)
    e = floor_log2(x)
    min_normal = 1 - bias
    if e < min_normal:
        unit = Fraction(2) ** (min_normal - mb)
        q = round_nearest_even(x / unit)
        if q >= (1 << mb):
            return (sign << (total - 1)) | (1 << mb)
        return (sign << (total - 1)) | q
    q = round_nearest_even(x / Fraction(2) ** (e - mb))
    if q == 1 << (mb + 1):
        q >>= 1
        e += 1
    exp = e + bias
    if exp >= maxexp:
        return inf
    return (sign << (total - 1)) | (exp << mb) | (q - (1 << mb))


def expected_bits(op, a, b, eb, mb, bias):
    total = 1 + eb + mb
    sign_bit = 1 << (total - 1)
    quiet_nan = ((1 << eb) - 1) << mb | (1 << (mb - 1))
    va = decode(a, eb, mb, bias)
    vb = decode(b, eb, mb, bias)
    if va is None or vb is None:
        return quiet_nan
    if op == "+":
        r = va + vb
    elif op == "-":
        r = va - vb
    elif op == "*":
        r = va * vb
    else:
        if vb == 0:
            if va == 0:
                return quiet_nan
            return (sign_bit if va < 0 else 0) | (((1 << eb) - 1) << mb)
        r = va / vb
    return encode(r, eb, mb, bias)


def numpy_f16(op, a, b):
    fa = np.frombuffer(struct.pack("<H", a), dtype=np.float16)[0]
    fb = np.frombuffer(struct.pack("<H", b), dtype=np.float16)[0]
    with np.errstate(all="ignore"):
        if op == "+":
            r = np.float16(fa + fb)
        elif op == "-":
            r = np.float16(fa - fb)
        elif op == "*":
            r = np.float16(fa * fb)
        else:
            r = np.float16(fa / fb)
    return struct.unpack("<H", np.float16(r).tobytes())[0]


def probe_pairs(eb, mb, bias):
    """Boundary pairs: values on or next to a rounding edge."""
    sign_bit = 1 << (eb + mb)
    one = bias << mb
    three = (bias + 1) << mb
    five = (bias + 2) << mb
    half = (bias - 1) << mb
    largest = (((1 << eb) - 2) << mb) | ((1 << mb) - 1)
    smallest_normal = 1 << mb
    max_subnormal = (1 << mb) - 1
    cases = []
    for delta in (0, 1, 2, -1, -2):
        cases.append((one, one + delta))
    cases.append((largest, largest))
    cases.append((largest, one))
    cases.append((largest, half))
    cases.append((smallest_normal, half))
    cases.append((smallest_normal, one))
    cases.append((max_subnormal, one))
    cases.append((1, one))
    cases.append((1, 1))
    cases.append((max_subnormal, 1))
    cases.append((one, three))
    cases.append(((bias + 1) << mb, three))
    cases.append((one, five))
    cases.append((one + 1, (bias + 1) << mb))
    cases.append((sign_bit | one, one))
    cases.append((sign_bit | one, sign_bit | one))
    cases.append((sign_bit | ((bias + 1) << mb), one))
    return cases


def random_pairs(n, upper, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        a = rng.randrange(0, upper)
        b = rng.randrange(0, upper)
        if rng.random() < 0.5:
            a |= 1 << 15
        if rng.random() < 0.5:
            b |= 1 << 15
        out.append((a, b))
    return out


def emit(name, pairs, eb, mb, bias, out):
    out.write(f"pub let {name} : Array[FpVector] = [\n")
    for a, b in pairs:
        s = expected_bits("+", a, b, eb, mb, bias)
        d = expected_bits("-", a, b, eb, mb, bias)
        m = expected_bits("*", a, b, eb, mb, bias)
        q = expected_bits("/", a, b, eb, mb, bias)
        out.write(
            f"  {{ a: {a}, b: {b}, sum: {s}, difference: {d},"
            f" product: {m}, quotient: {q} }},\n"
        )
    out.write("]\n\n")


def main():
    f16_pairs = probe_pairs(5, 10, 15) + random_pairs(150, 0x7C00, 20261126)
    bf16_pairs = probe_pairs(8, 7, 127) + random_pairs(150, 0x7F80, 20261127)

    out = []
    import io

    out = io.StringIO()

    out.write("// Generated by tools/gen_fp_vectors.py -- do not edit by hand.\n")
    out.write("// Reference side: exact rational arithmetic, one rounding per result.\n")
    out.write(
        f"// {len(f16_pairs)} binary16 pairs and {len(bf16_pairs)} bfloat16 pairs.\n\n"
    )
    out.write("pub struct FpVector {\n")
    out.write("  a : Int\n")
    out.write("  b : Int\n")
    out.write("  sum : Int\n")
    out.write("  difference : Int\n")
    out.write("  product : Int\n")
    out.write("  quotient : Int\n")
    out.write("}\n\n")
    emit("f16_arithmetic_vectors", f16_pairs, 5, 10, 15, out)
    emit("bf16_arithmetic_vectors", bf16_pairs, 8, 7, 127, out)
    print(out.getvalue(), end="")

    agree = 0
    differ = 0
    diffs = []
    for a, b in f16_pairs:
        for op in ("+", "-", "*", "/"):
            want = expected_bits(op, a, b, 5, 10, 15)
            got = numpy_f16(op, a, b)
            if got == want:
                agree += 1
            else:
                differ += 1
                diffs.append((op, a, b, got, want))
    print(f"// numpy float16: {agree} agreement, {differ} disagreement")
    for op, a, b, got, want in diffs:
        print(f"//   {op} {a:#06x} {b:#06x}: numpy {got:#06x}, exact {want:#06x}")


if __name__ == "__main__":
    main()
