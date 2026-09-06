#!/usr/bin/env python3
"""Independent deterministic oracle and public-surface checks for QWN-031B."""

from pathlib import Path
import math
import struct
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/quant/reference_codec.seen"
CONTRACT = ROOT / "docs/qwen-reference-codec-contract.md"
GROUP = 64


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f16_bits(value: float) -> int:
    return int.from_bytes(struct.pack("<e", value), "little")


def f16_value(bits: int) -> float:
    return struct.unpack("<e", bits.to_bytes(2, "little"))[0]


def encode_q8(values: list[float], row_elements: int) -> tuple[bytes, list[int]]:
    if row_elements <= 0 or len(values) % row_elements:
        raise ValueError("geometry")
    codes = bytearray()
    scales: list[int] = []
    for row in range(0, len(values), row_elements):
        for start in range(row, row + row_elements, GROUP):
            end = min(start + GROUP, row + row_elements)
            group = [f32(value) for value in values[start:end]]
            if not all(math.isfinite(value) for value in group):
                raise ValueError("input")
            maximum = max(map(abs, group), default=0.0)
            if maximum == 0.0:
                scales.append(0)
                codes.extend(b"\0" * len(group))
                continue
            scale = f32(maximum / 127.0)
            try:
                stored = f16_bits(scale)
            except OverflowError as error:
                raise ValueError("range") from error
            if stored == 0 or not math.isfinite(f16_value(stored)):
                raise ValueError("range")
            scales.append(stored)
            for value in group:
                quantized = max(-127, min(127, round(f32(value / scale))))
                codes.append(quantized & 0xFF)
    return bytes(codes), scales


class Q8ReferenceCodecTests(unittest.TestCase):
    def test_exact_ties_signed_range_and_scale(self) -> None:
        values = [127.0, -127.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5]
        values.extend([0.0] * (GROUP - len(values)))
        codes, scales = encode_q8(values, GROUP)
        self.assertEqual(scales, [0x3C00])
        self.assertEqual(list(codes[:8]), [127, 129, 0, 2, 2, 0, 254, 254])
        self.assertNotIn(128, codes)

    def test_rows_tail_and_repeat_are_canonical(self) -> None:
        values = [0.0] * 130
        values[0], values[64], values[65], values[129] = 127.0, 2.0, 4.0, -3.0
        first = encode_q8(values, 65)
        second = encode_q8(values, 65)
        self.assertEqual(first, second)
        codes, scales = first
        self.assertEqual(len(codes), 130)
        self.assertEqual(len(scales), 4)
        self.assertEqual([codes[i] for i in (0, 64, 65, 129)], [127, 127, 127, 129])

    def test_zero_nonfinite_and_scale_range(self) -> None:
        self.assertEqual(encode_q8([0.0, -0.0], 2), (b"\0\0", [0]))
        with self.assertRaisesRegex(ValueError, "input"):
            encode_q8([math.inf], 1)
        with self.assertRaisesRegex(ValueError, "input"):
            encode_q8([math.nan], 1)
        with self.assertRaisesRegex(ValueError, "range"):
            encode_q8([1.0e-9], 1)
        with self.assertRaisesRegex(ValueError, "range"):
            encode_q8([1.0e7], 1)

    def test_native_surface_and_contract_are_explicit(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        contract = CONTRACT.read_text(encoding="utf-8")
        for spelling in (
            "ReferenceQ8Buffer",
            "encodeQ8SymG64Buffer",
            "decodeQ8SymG64Buffer",
            'codec: "Q8_SYM_G64"',
            "qwen.codec.geometry",
            "quantized == -128",
        ):
            self.assertIn(spelling, source)
        for rule in (
            "max_abs / 127",
            "round-to-nearest",
            "ties-to-even",
            "Code `-128` is never emitted",
            "stores no padding bytes",
            "float(code) * float(fp16_scale)",
        ):
            self.assertIn(rule, contract)


if __name__ == "__main__":
    unittest.main()
