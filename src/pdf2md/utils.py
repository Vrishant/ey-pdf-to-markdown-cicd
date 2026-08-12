# Location: src/pdf2md/utils.py
from PIL import Image

# Reference: a full-page-width table crop at 300 DPI is roughly this area.
# Small crops (a 3x3 table) don't need the same token budget as a dense
# full-page table — scaling avoids paying for unused max_new_tokens headroom
# on every call.
_REFERENCE_AREA_PX = 300 * 300 * 4  # ~ a quarter-page crop at 300 DPI
_MIN_TOKENS = 256


def estimate_max_tokens(crop: Image.Image, base_tokens: int) -> int:
    area = crop.width * crop.height
    scale = min(1.0, area / _REFERENCE_AREA_PX)
    return max(_MIN_TOKENS, int(base_tokens * scale))