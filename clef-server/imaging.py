"""Image decoding / normalisation (Pillow only)."""
from __future__ import annotations

import io
import os
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_SIDE = int(os.environ.get("MAX_IMAGE_SIDE", "1024"))
# Guard against decompression bombs (~ 16k x 16k)
Image.MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", str(16384 * 16384)))


class BadImage(ValueError):
    pass


def load_image(data: bytes, max_side: int = MAX_SIDE) -> Image.Image:
    """Decode bytes -> RGB PIL image, EXIF-rotated, longest side <= max_side."""
    if not data:
        raise BadImage("empty upload")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            probe = Image.open(io.BytesIO(data))
            probe.verify()  # cheap integrity check; invalidates `probe`
            img = Image.open(io.BytesIO(data))
            img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
        raise BadImage(f"could not decode image: {e}") from e

    try:
        img = ImageOps.exif_transpose(img)
    except Exception:  # broken EXIF should not kill the request
        pass

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        img = bg
    else:
        img = img.convert("RGB")

    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img
