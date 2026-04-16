"""Default filesystem paths used across the hue_map package.

All paths are absolute and derived from the location of this file, so the
package works the same way whether it's run as ``py -m hue_map.pipeline``
from the repo root or from inside ``src/``.
"""
from __future__ import annotations

from pathlib import Path


# Package locations.
PACKAGE_ROOT = Path(__file__).resolve().parent         # .../src/hue_map
SRC_ROOT = PACKAGE_ROOT.parent                         # .../src
REPO_ROOT = SRC_ROOT.parent                            # .../Anvil-CLIP

# User-facing directories inside src/.
DEFAULT_INPUT_DIR = SRC_ROOT / "input"                 # drop screenshots here
DEFAULT_REFERENCES_DIR = SRC_ROOT / "references"       # reference library
DEFAULT_OUTPUT_DIR = SRC_ROOT / "output"               # generated reports

# Default source for the populate script: the flat-PNG folder produced by
# ``download_anvil_items.py`` in the repo root.
DEFAULT_SOURCE_PNGS_DIR = REPO_ROOT / "itemlist"

# Image types we accept both as queries and as references.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}