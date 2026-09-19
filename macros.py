"""
Global constants and helpers shared by several Mix-And-Match files.

Constants used by only one file are declared at the top of that file instead.
"""

import os
from pathlib import Path

# Folder of the project; outputs and the prompt cache live under it.
PROJECT_DIR = Path(__file__).resolve().parent

# Where the results of every run are written.
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Side of one PixelDiT patch token in pixels. Image sizes and crop boundaries must be multiples of it.
PATCH_SIZE_PIXELS = 16

# Size of every generated image in pixels (the resolution the released PixelDiT checkpoint was last trained on).
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024

# Denoising steps of every run.
NUM_STEPS = 70

# Tokens per prompt fed to the transformer (PixelDiT's model_max_length).
PROMPT_LENGTH = 300

# Environment variable that turns the page cache dropping below off, for a machine that would rather keep it.
KEEP_PAGE_CACHE_VARIABLE = "MIX_N_MATCH_KEEP_PAGE_CACHE"

# Whether files are dropped from the page cache once they have been written or read. Reading the checkpoint and
# writing a folder run's images both fill the page cache, and WSL2 never hands that memory back to Windows, so the
# VM's footprint grows until the host starves; dropping a file's pages lets the next file reuse them instead.
# posix_fadvise exists on Linux (WSL included) but not on Windows or macOS.
DROP_PAGE_CACHE = hasattr(os, "posix_fadvise") and os.environ.get(KEEP_PAGE_CACHE_VARIABLE, "") != "1"


def drop_page_cache(path):
    """
    Drops the pages the kernel cached for a file; does nothing when DROP_PAGE_CACHE is off.

    The file itself is untouched: this only tells the kernel that its cached copy is no longer worth keeping, so
    reading the file again costs a disk read.

    Args:
        path: Path of the file whose cached pages are dropped.
    """
    if not DROP_PAGE_CACHE:
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)  # pages still dirty cannot be dropped
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(descriptor)


def save_image(image, path):
    """
    Saves a PIL image and drops the pages the kernel cached for it (see drop_page_cache).

    Args:
        image: PIL image to save.
        path: Path of the file to write.
    """
    image.save(path)
    drop_page_cache(path)
