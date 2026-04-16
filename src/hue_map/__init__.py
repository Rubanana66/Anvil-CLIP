"""Hue-Map: HSV color-histogram item recognition for Anvil Empires stockpiles.

Package layout
--------------
``paths``
    Default filesystem locations (input/, references/, output/).
``slots``
    Screenshot loading + slot detection/cropping. Keeps the full bottom of
    each slot so stack-count pixels are included when histogramming.
``color_matcher``
    HSV histogram matching using the Bhattacharyya coefficient. Detects the
    slot background per query (so we don't hardcode a global color) and
    masks transparent/near-black pixels on references.
``reporting``
    Renders a self-contained HTML report (all images base64-embedded).
``pipeline``
    Orchestrates load image -> detect slots -> classify -> write report.
``populate``
    Builds the ``references/`` hierarchy from a flat folder of PNGs
    (one file per item name), as produced by the upstream wiki scraper.

Typical use:
    ``py -m hue_map.populate``  to (re)populate references/
    ``py -m hue_map.pipeline``  to process everything in input/
"""

__version__ = "0.1.0"
