"""Decode an upload into the RGB image the model sees (Pillow only)."""

import io
import os
from contextlib import suppress

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_SIDE = int(os.environ.get("MAX_IMAGE_SIDE", "1024"))
MAX_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", str(16384 * 16384)))  # decompression-bomb guard
Image.MAX_IMAGE_PIXELS = MAX_PIXELS  # Pillow's own hard stop (at 2x) stays as a second line of defence

_DECODE_ERRORS = (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError)


class BadImage(ValueError):
    """The upload is not a decodable image (→ HTTP 400)."""


def load_image(data: bytes, max_side: int = MAX_SIDE) -> Image.Image:
    """Decode bytes → RGB image, EXIF-rotated, transparency flattened onto white, longest side ≤ max_side."""
    if not data:
        raise BadImage("empty upload")
    try:
        img = Image.open(io.BytesIO(data))  # reads the header only
        if img.width * img.height > MAX_PIXELS:
            raise BadImage(f"image too large: {img.width}x{img.height} pixels")
        if max_side:
            # JPEG: let libjpeg decode at 1/2, 1/4 or 1/8 scale (never below max_side) instead of
            # decoding a full-size phone photo only to shrink it. A no-op for other formats.
            img.draft(None, (max_side, max_side))
        img.load()
    except BadImage:
        raise
    except _DECODE_ERRORS as e:
        raise BadImage(f"could not decode image: {e}") from e

    with suppress(Exception):  # broken EXIF should not fail the request
        ImageOps.exif_transpose(img, in_place=True)

    img = _to_rgb(img)
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img  # convert() would make a needless full copy
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba)  # an RGBA mask uses its alpha channel
        return flat
    return img.convert("RGB")
