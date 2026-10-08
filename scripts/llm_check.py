"""Check the LLM configuration in llm-server/.env (used by ./run.sh llm).

python scripts/llm_check.py            # reachability + key check (free)
python scripts/llm_check.py --classify # also send one small test picture (costs ~nothing)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from PIL import Image, ImageDraw

LLM_DIR = Path(__file__).resolve().parents[1] / "llm-server"
sys.path.insert(0, str(LLM_DIR))

from llm_client import LLMClient, LLMConfig, UpstreamError, load_dotenv  # noqa: E402  # needs the path above


def test_picture() -> Image.Image:
    """A crude cartoon hot dog: enough for a vision model to have an opinion."""
    im = Image.new("RGB", (512, 320), (250, 244, 230))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([60, 110, 452, 230], radius=60, fill=(225, 170, 95))  # bun
    d.rounded_rectangle([40, 140, 472, 200], radius=30, fill=(170, 60, 40))  # sausage
    for x in range(90, 430, 40):  # mustard zig-zag
        d.line([x, 160, x + 20, 180, x + 40, 160], fill=(240, 200, 30), width=6)
    return im


async def try_classify(client: LLMClient) -> bool:
    print("  … sending a test picture of a hot dog")
    try:
        v = await client.classify(test_picture())
    except UpstreamError as e:
        print(f"  ✗ {e}")
        return False
    verdict = "hotdog" if v.label == "hotdog" else "not hotdog"
    confidence = v.probabilities[v.label]
    print(f"  ✓ the model says: {verdict} ({confidence:.0%}, {v.confidence_source}) in {v.latency_ms / 1000:.1f} s")
    if v.reason:
        print(f"    “{v.reason}”")
    return True


async def run(classify: bool) -> int:
    cfg = LLMConfig.from_env()
    print(f"  provider: {cfg.base_url}")
    print(f"  model:    {cfg.model}")
    async with LLMClient(cfg) as client:
        ok, detail = await client.ping()
        if not ok:
            print(f"  ✗ {detail}")
            return 1
        print("  ✓ API reachable")
        if classify and not await try_classify(client):
            return 1
    return 0


if __name__ == "__main__":
    load_dotenv(LLM_DIR / ".env")
    sys.exit(asyncio.run(run("--classify" in sys.argv)))
