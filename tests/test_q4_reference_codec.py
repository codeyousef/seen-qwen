#!/usr/bin/env python3
"""Independent Q4_SYM_G64 byte oracle and contract checks for QWN-031C."""

from pathlib import Path
import math
import struct
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/quant/reference_codec.seen"
CONTRACT = ROOT / "docs/qwen-reference-codec-contract.md"


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def encode(values: list[float], width: int) -> tuple[bytes, list[int]]:
    if width <= 0 or len(values) % width:
        raise ValueError("geometry")
    payload = bytearray(); scales: list[int] = []
    for row in range(0, len(values), width):
        for start in range(row, row + width, 64):
            group = [f32(v) for v in values[start:min(start + 64, row + width)]]
            if not all(math.isfinite(v) for v in group):
                raise ValueError("input")
            maximum = max(map(abs, group), default=0.0)
            scale = f32(maximum / 7.0) if maximum else 0.0
            try:
                bits = int.from_bytes(struct.pack("<e", scale), "little")
            except OverflowError as error:
                raise ValueError("range") from error
            if maximum and bits == 0:
                raise ValueError("range")
            scales.append(bits)
            codes = [max(-8, min(7, round(f32(v / scale)))) if maximum else 0
                     for v in group]
            for index in range(0, len(codes), 2):
                high = codes[index + 1] if index + 1 < len(codes) else 0
                payload.append((codes[index] & 15) | ((high & 15) << 4))
    return bytes(payload), scales


class Q4ReferenceCodecTests(unittest.TestCase):
    def test_exact_ties_and_low_first_packing(self) -> None:
        values = [7.0, -7.0, .5, 1.5, 2.5, -.5, -1.5, -2.5] + [0.0] * 56
        payload, scales = encode(values, 64)
        self.assertEqual(scales, [0x3C00])
        self.assertEqual(payload[:4], bytes([0x97, 0x20, 0x02, 0xEE]))

    def test_row_tails_repeat_and_pad_independently(self) -> None:
        values = [0.0] * 130
        values[0], values[64], values[65], values[129] = 7.0, 2.0, 4.0, -3.0
        first = encode(values, 65)
        self.assertEqual(first, encode(values, 65))
        self.assertEqual(len(first[0]), 66)
        self.assertEqual(len(first[1]), 4)
        self.assertEqual(first[0][32], 7)
        self.assertEqual(first[0][65], 9)

    def test_source_and_docs_freeze_the_normative_surface(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        contract = CONTRACT.read_text(encoding="utf-8")
        for text in ("ReferenceQ4Buffer", "encodeQ4SymG64Buffer",
                     "decodeQ4SymG64Buffer", 'codec: "Q4_SYM_G64"'):
            self.assertIn(text, source)
        for text in ("max_abs / 7", "[-8, 7]", "low nibble",
                     "zero high padding nibble", "float(code) * float(fp16_scale)"):
            self.assertIn(text, contract)


if __name__ == "__main__":
    unittest.main()
