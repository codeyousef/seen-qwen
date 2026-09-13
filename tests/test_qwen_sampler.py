"""Independent deterministic oracle for the QWN-044B host sampler."""

import math
import struct
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MASK64 = (1 << 64) - 1
DEFAULT_SEED = 0x9E3779B97F4A7C15
MULTIPLIER = 0x2545F4914F6CDD1D


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


class Rng:
    def __init__(self, seed: int) -> None:
        self.state = seed or DEFAULT_SEED
        self.draws = 0

    def next24(self) -> int:
        value = self.state
        value ^= value >> 12
        value ^= (value << 25) & MASK64
        value ^= value >> 27
        self.state = value & MASK64
        self.draws += 1
        return ((self.state * MULTIPLIER) & MASK64) >> 40


def sample(logits, history, *, temperature=1.0, top_p=1.0, top_k=None,
           min_p=0.0, repetition_penalty=1.0, presence_penalty=0.0,
           rng):
    """Formula oracle, deliberately independent of the Seen heap sort."""
    if top_k is None:
        top_k = len(logits)
    present = set(history)
    scores = []
    for token, original in enumerate(logits):
        score = f32(original)
        if token in present:
            score = f32(score * repetition_penalty if score < 0.0
                        else score / repetition_penalty)
            score = f32(score - presence_penalty)
        scores.append(f32(score / temperature))

    ranked = sorted(range(len(scores)), key=lambda token: (-scores[token], token))
    kth = scores[ranked[top_k - 1]]
    kept = {token for token, score in enumerate(scores) if score >= kth}
    maximum = scores[ranked[0]]
    weights = [math.exp(score - maximum) for score in scores]

    removable = (1.0 - top_p) * sum(weights[token] for token in kept)
    removed = 0.0
    for token in reversed(ranked):
        if len(kept) == 1:
            break
        if token in kept and removed + weights[token] <= removable:
            kept.remove(token)
            removed += weights[token]

    if min_p > 0.0:
        best = ranked[0]
        kept = {token for token in kept
                if token == best or weights[token] >= min_p}

    total = sum(weights[token] for token in kept)
    threshold = rng.next24() / 16777216.0 * total
    cumulative = 0.0
    for token in range(len(scores)):
        if token in kept:
            cumulative += weights[token]
            if cumulative > threshold:
                return token
    return min(kept)


class QwenSamplerOracleTest(unittest.TestCase):
    def test_locked_rng_vectors(self) -> None:
        rng = Rng(20260827)
        self.assertEqual([rng.next24() for _ in range(8)], [
            12895245, 5889182, 7097424, 733511,
            14913692, 5117147, 9045593, 12379888,
        ])

    def test_filter_and_penalty_vectors(self) -> None:
        uniform_rng = Rng(1)
        self.assertEqual([
            sample([0.0] * 4, [], rng=uniform_rng) for _ in range(8)
        ], [1, 2, 2, 1, 0, 3, 3, 2])

        tie_rng = Rng(1)
        self.assertEqual([
            sample([5.0, 4.0, 4.0, 0.0, -1.0], [], top_k=2,
                   rng=tie_rng) for _ in range(8)
        ], [0, 1, 1, 0, 0, 1, 2, 1])

        penalty_rng = Rng(9)
        self.assertEqual(sample([2.0, 1.8, -1.0], [0, 0], top_p=0.01,
                                top_k=1, repetition_penalty=2.0,
                                presence_penalty=0.5, rng=penalty_rng), 1)

    def test_source_contract_is_seen_owned_and_bounded(self) -> None:
        source = (ROOT / "src/model/sampling.seen").read_text()
        for required in (
            "QWEN_SAMPLER_MAX_VOCABULARY", "248320 as UInt64",
            "QWEN_SAMPLER_MAX_HISTORY", "262144 as UInt64",
            "value = value ^ (value >> 12)",
            "value = value ^ (value << 25)",
            "value = value ^ (value >> 27)",
            "policy.repetitionPenalty", "policy.presencePenalty",
            "policy.temperature", "policy.topK", "policy.topP", "policy.minP",
            "scores.free(); kept.free(); ids.free()",
        ):
            self.assertIn(required, source)
        for forbidden in ("random(", "rand(", "cuda", "fallback"):
            self.assertNotIn(forbidden, source.lower())

    def test_runner_is_hard_scoped_and_uses_exact_release(self) -> None:
        runner = (ROOT / "scripts/cuda/run_qwn_044b.sh").read_text()
        for required in (
            "scripts/oracle/run_bounded.sh", "QWN_TASKS_MAX=32",
            "v0.20.8/extracted/seen-0.20.8-linux-x64",
            "76833346fbe3e01cda0aeb2b34d585a2115086ccee3e87205059b055d22ac2b6",
            "--jobs 1 --opt-jobs 1 --no-fork", "--sanitize undefined",
            "qwn_044a.sh", "compute-sanitizer", "toolchain_hash_after",
            "outside_objects_after",
        ):
            if required == "compute-sanitizer":
                self.assertIn(required, (ROOT / "scripts/cuda/run_qwn_044a.sh").read_text())
            else:
                self.assertIn(required, runner)
        self.assertNotIn("/usr/local/bin/seen", runner)
        self.assertNotIn("sudo", runner)


if __name__ == "__main__":
    unittest.main()
