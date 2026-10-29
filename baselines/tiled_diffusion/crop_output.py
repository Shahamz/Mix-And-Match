"""
Writing the tiled-diffusion baseline's images and its run metadata.

Two kinds of image:

    tiles/   one PNG per candidate tile, at the size it was generated at
    combos/  one PNG per way of picking a candidate for every crop, the chosen tiles in the crops' grid

The combinations are the point of the experiment: if the candidates of a crop really are interchangeable, every
combination should show the same seams.

Copied from tiled-diffusion's crop_output.py; the resize on paste is gone, because every tile is now generated at
exactly the size it is placed at.
"""

import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image

from macros import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_STEPS, save_image

from .crop_layout import canvas_size, grid_shape

# Colour the combination canvas starts as, so any area no crop covers (an empty grid cell) is obvious.
UNCOVERED_COLOUR = (0, 0, 0)


def to_image(latent):
    """
    Args:
        latent: a LatentClass whose run has finished, holding its image as an HWC float array in [0, 1].

    Returns:
        It as a PIL RGB image.
    """
    return Image.fromarray((np.asarray(latent.image) * 255.0).clip(0, 255).astype(np.uint8))


def save_tiles(latents_arr, placed_crops, out_dir):
    """
    Writes every candidate tile, named by its crop's index in the config.

    Args:
        latents_arr: the finished tiles.
        placed_crops: the crops from crop_layout.build_latents.
        out_dir: Path of the run folder.

    Returns:
        The paths written.
    """
    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for placed in placed_crops:
        for candidate_index, flat_index in enumerate(placed.tile_indices):
            path = tiles_dir / f"crop{placed.index:02d}_cand{candidate_index:02d}.png"
            save_image(to_image(latents_arr[flat_index]), path)
            written.append(path)

    print(f"Wrote {len(written)} tiles to {tiles_dir}")
    return written


def compose_combination(latents_arr, placed_crops, choice):
    """
    Builds one full size image from one candidate per crop.

    Args:
        latents_arr: the finished tiles.
        placed_crops: the crops from crop_layout.build_latents.
        choice: the candidate index to use for each crop, in the order of placed_crops.

    Returns:
        PIL RGB image of the whole grid, crop_layout.canvas_size.
    """
    canvas = Image.new("RGB", canvas_size(len(placed_crops)), UNCOVERED_COLOUR)
    for placed, candidate_index in zip(placed_crops, choice, strict=True):
        canvas.paste(to_image(latents_arr[placed.tile_indices[candidate_index]]), (placed.place_x, placed.place_y))
    return canvas


def save_all_combinations(latents_arr, placed_crops, out_dir, max_combos=0):
    """
    Writes one full size image per combination of candidates.

    There are tiles_per_crop ** num_crops of them, which grows fast, so max_combos caps how many are written.

    Args:
        latents_arr: the finished tiles.
        placed_crops: the crops from crop_layout.build_latents.
        out_dir: Path of the run folder.
        max_combos: cap on how many are written; 0 means all of them.

    Returns:
        The paths written.
    """
    combos_dir = out_dir / "combos"
    combos_dir.mkdir(parents=True, exist_ok=True)

    candidates_per_crop = [len(placed.tile_indices) for placed in placed_crops]
    total = 1
    for count in candidates_per_crop:
        total *= count

    width, height = canvas_size(len(placed_crops))
    print(f"{total} combination(s) of candidates ({' x '.join(str(count) for count in candidates_per_crop)}), "
          f"each {width}x{height}")
    if max_combos and total > max_combos:
        print(f"  --max-combos is {max_combos}, so only the first {max_combos} will be written")

    written = []
    for choice in itertools.product(*[range(count) for count in candidates_per_crop]):
        if max_combos and len(written) >= max_combos:
            break
        name = "combo_" + "-".join(f"{candidate_index:02d}" for candidate_index in choice) + ".png"
        path = combos_dir / name
        save_image(compose_combination(latents_arr, placed_crops, choice), path)
        written.append(path)

    print(f"Wrote {len(written)} combination image(s) to {combos_dir}")
    return written


def save_run_meta(config, args, placed_crops, out_dir):
    """
    Dumps the settings, prompts and sizes actually used, next to the images.

    Args:
        config: MixNMatchConfig.
        args: the parsed command line.
        placed_crops: the crops from crop_layout.build_latents.
        out_dir: Path of the run folder.

    Returns:
        Path of the file written.
    """
    path = out_dir / "run_meta.json"
    width, height = canvas_size(len(placed_crops))
    rows, columns = grid_shape(len(placed_crops))
    meta = {
        # Paths are not JSON values, so they are written as text; everything else keeps its type.
        "args": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "seed": config.seed,
        "cfg_scale": config.cfg_scale,
        "num_steps": NUM_STEPS,
        "image_width": width,
        "image_height": height,
        "grid_rows": rows,
        "grid_columns": columns,
        "num_crops": config.num_crops,
        "tiles_per_crop": config.tiles_per_crop,
        "tile_prompts": config.tile_prompts,
        "tile_negative_prompts": config.tile_negative_prompts,
        "crops": [
            {
                "config_index": placed.index,
                "grid_position": placed.position,
                "grid_row": placed.row, "grid_column": placed.column,
                "placed_x": placed.place_x, "placed_y": placed.place_y,
                "placed_width": IMAGE_WIDTH, "placed_height": IMAGE_HEIGHT,
            }
            for placed in placed_crops
        ],
    }
    path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {path}")
    return path
