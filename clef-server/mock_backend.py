"""Deterministic mock classifier. Imports nothing heavy (no torch)."""
from __future__ import annotations

import hashlib


class MockBackend:
    name = "mock"
    device = "mock"
    ready = True
    error = None

    def start(self) -> None:  # nothing to load
        pass

    def classify(self, image, raw: bytes) -> tuple[dict[str, float], float]:
        h = hashlib.sha256(raw).digest()
        u = int.from_bytes(h[:8], "big") / 2**64          # [0,1)
        v = int.from_bytes(h[8:16], "big") / 2**64
        # Plausible, mostly confident outputs: ~35% hotdog, skewed to extremes.
        is_hd = u < 0.35
        conf = 0.70 + 0.295 * (v ** 0.5)                  # 0.70 .. 0.995
        p_hd = conf if is_hd else 1.0 - conf
        p_hd = round(p_hd, 6)
        latency_ms = round(180.0 + 420.0 * ((h[16] << 8 | h[17]) / 65535), 1)
        return {"hotdog": p_hd, "not_hotdog": round(1.0 - p_hd, 6)}, latency_ms
