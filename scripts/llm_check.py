"""Check the LLM configuration in llm-server/.env (used by ./run.sh llm).

    python scripts/llm_check.py            # reachability + key check (free)
    python scripts/llm_check.py --classify # also send one small test picture (costs ~nothing)
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "llm-server"))
os.chdir(ROOT / "llm-server")

from llm_client import LLMClient, LLMConfig, UpstreamError, load_dotenv  # noqa: E402

load_dotenv(ROOT / "llm-server" / ".env")


def test_picture():
    """A crude cartoon hot dog: enough for a vision model to have an opinion."""
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (512, 320), (250, 244, 230))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([60, 110, 452, 230], radius=60, fill=(225, 170, 95))    # bun
    d.rounded_rectangle([40, 140, 472, 200], radius=30, fill=(170, 60, 40))     # sausage
    for x in range(90, 430, 40):                                                # mustard zig-zag
        d.line([x, 160, x + 20, 180, x + 40, 160], fill=(240, 200, 30), width=6)
    return im


async def run(classify: bool) -> int:
    cfg = LLMConfig.from_env()
    print(f"  provider: {cfg.base_url}")
    print(f"  model:    {cfg.model}")
    client = LLMClient(cfg)
    try:
        ok, detail = await client.ping()
        if not ok:
            print(f"  ✗ {detail}")
            return 1
        print("  ✓ API reachable")
        if classify:
            print("  … sending a test picture of a hot dog")
            try:
                r = await client.classify(test_picture())
            except UpstreamError as e:
                print(f"  ✗ {e}")
                return 1
            p = r["probabilities"]["hotdog"]
            verdict = "hotdog" if p >= 0.5 else "not hotdog"
            print(f"  ✓ the model says: {verdict} ({max(p, 1 - p):.0%}, {r['confidence_source']}) "
                  f"in {r['latency_ms'] / 1000:.1f} s")
            if r.get("reason"):
                print(f"    “{r['reason']}”")
        return 0
    finally:
        await client.aclose()


if __name__ == "__main__":
    sys.exit(asyncio.run(run("--classify" in sys.argv)))
