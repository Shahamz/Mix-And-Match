"""
Loads and validates Mix-And-Match config JSON files.

A config describes one run: the seed, the guidance scale, the number of crops and tiles, one positive plus one
negative prompt for the background and for every tile, when the tile prompts start (late_prompting), how the crops
are chosen, and what is saved. Everything else about the method is fixed in the code.

`load_config` returns a `MixNMatchConfig` or raises `ConfigError` with a readable message. For a standard run
(generate.py --std) only the fields in STANDARD_RUN_FIELDS are read; the tile and crop fields are ignored.
`build_crop_map` validates a static crop layout and turns it into a patch-grid map of crop indices.
"""

import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from macros import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_STEPS, PATCH_SIZE_PIXELS

# Keys every crop entry must have (pixels; x, y is the top-left corner).
CROP_KEYS = ("x", "y", "width", "height")

# The fields a standard run (generate.py --std) requires; all tile and crop fields are ignored.
STANDARD_RUN_FIELDS = ("prefix", "seed", "cfg_scale", "background_prompt", "background_negative_prompt",
                       "sequential_cfg")

# Expected JSON type of every scalar field (prompt grids and crops are checked separately).
SCALAR_FIELD_TYPES = {
    "prefix": str,
    "seed": int,
    "cfg_scale": (int, float),
    "num_crops": int,
    "tiles_per_crop": int,
    "background_prompt": str,
    "background_negative_prompt": str,
    "late_prompting": int,
    "static_cropping": bool,
    "sequential_cfg": bool,
    "visualize": bool,
    "save_separate_tiles": bool,
    "save_saliency_maps": bool,
}


class ConfigError(Exception):
    """Raised when a config file is missing, malformed, or inconsistent."""


@dataclass
class MixNMatchConfig:
    """
    All settings of one Mix-And-Match run.
    In a standard run, tile and crop fields missing from the JSON are None.
    """

    prefix: str  # name of the run's outputs
    seed: int  # seed of the initial noise
    cfg_scale: float  # classifier-free guidance scale
    num_crops: int  # n: number of tile crops
    tiles_per_crop: int  # k: tiles per crop, each with its own prompt
    background_prompt: str
    background_negative_prompt: str
    tile_prompts: list  # num_crops lists of tiles_per_crop strings
    tile_negative_prompts: list  # same shape as tile_prompts
    late_prompting: int  # step at which the image is split into tiles and the tile prompts start
    static_cropping: bool  # true: the crops come from "crops"; false: from the saliency of the tile prompts
    sequential_cfg: bool  # run the negative and positive guidance passes one after the other (less memory)
    visualize: bool  # save sheets with every tile combination
    save_separate_tiles: bool  # save every tile as its own image (needed by the viewer)
    save_saliency_maps: bool  # save the saliency heat maps and the crop segmentation (dynamic cropping only)
    crops: list | None = None  # list of {"x", "y", "width", "height"} dicts, used only when static_cropping is true


def build_crop_map(crops, width, height):
    """
    Validates a crop layout and maps every patch of the image to the crop that covers it.

    Crops must be dicts with integer x, y, width, height (pixels, multiples of PATCH_SIZE_PIXELS), lie inside the
    image, not overlap, and together cover the whole image.

    Args:
        crops: list of {"x", "y", "width", "height"} dicts.
        width: image width in pixels.
        height: image height in pixels.

    Returns:
        np.ndarray of shape [height // PATCH_SIZE_PIXELS, width // PATCH_SIZE_PIXELS] holding crop indices.

    Raises:
        ConfigError: if the layout breaks any of the rules above.
    """
    crop_map = np.full((height // PATCH_SIZE_PIXELS, width // PATCH_SIZE_PIXELS), -1)
    for crop_index, crop in enumerate(crops):
        if not isinstance(crop, dict) or sorted(crop) != sorted(CROP_KEYS):
            raise ConfigError(f"crop {crop_index} must be an object with exactly the keys {list(CROP_KEYS)}")
        if any(type(crop[key]) is not int for key in CROP_KEYS):
            raise ConfigError(f"crop {crop_index}: x, y, width, height must be integers")
        if any(crop[key] % PATCH_SIZE_PIXELS for key in CROP_KEYS):
            raise ConfigError(f"crop {crop_index}: x, y, width, height must be multiples of {PATCH_SIZE_PIXELS} pixels")
        x, y, crop_width, crop_height = (crop[key] for key in CROP_KEYS)
        if crop_width <= 0 or crop_height <= 0:
            raise ConfigError(f"crop {crop_index}: width and height must be positive")
        if x < 0 or y < 0 or x + crop_width > width or y + crop_height > height:
            raise ConfigError(f"crop {crop_index} ({crop}) reaches outside the {width}x{height} image")

        region = crop_map[
            y // PATCH_SIZE_PIXELS : (y + crop_height) // PATCH_SIZE_PIXELS,
            x // PATCH_SIZE_PIXELS : (x + crop_width) // PATCH_SIZE_PIXELS,
        ]
        overlapped_crops = sorted(set(region[region >= 0].tolist()))
        if overlapped_crops:
            raise ConfigError(f"crop {crop_index} overlaps crop(s) {overlapped_crops}")
        region[:] = crop_index

    uncovered_rows, uncovered_cols = np.nonzero(crop_map < 0)
    if len(uncovered_rows):
        raise ConfigError(
            f"crops do not cover the whole image, e.g. the patch at x={uncovered_cols[0] * PATCH_SIZE_PIXELS}, "
            f"y={uncovered_rows[0] * PATCH_SIZE_PIXELS} is not in any crop"
        )
    return crop_map


def check_prompt_grid(field_name, prompt_grid, num_crops, tiles_per_crop):
    """
    Checks that a tile prompt field is a list of num_crops lists, each holding tiles_per_crop strings.

    Args:
        field_name: config field name, used in the error message.
        prompt_grid: the field's value from the JSON.
        num_crops: expected number of outer lists.
        tiles_per_crop: expected number of strings per inner list.

    Raises:
        ConfigError: if the shape or element types are wrong.
    """
    is_valid = (
        isinstance(prompt_grid, list)
        and len(prompt_grid) == num_crops
        and all(isinstance(crop_prompts, list) and len(crop_prompts) == tiles_per_crop for crop_prompts in prompt_grid)
        and all(isinstance(prompt, str) for crop_prompts in prompt_grid for prompt in crop_prompts)
    )
    if not is_valid:
        raise ConfigError(
            f'"{field_name}" must be a list of {num_crops} lists (one per crop), each with {tiles_per_crop} strings '
            "(one per tile)"
        )


def load_config(config_path, standard_run=False):
    """
    Reads a config JSON file and validates its fields.

    Args:
        config_path: path to the JSON file.
        standard_run: True for a standard run (generate.py --std): only STANDARD_RUN_FIELDS are required and
            validated; the tile and crop fields are ignored (they may be missing and are then None).

    Returns:
        MixNMatchConfig with the file's values.

    Raises:
        ConfigError: if the file is missing, is not valid JSON, has unknown fields, or a used field is missing or
            invalid.
    """
    config_path = Path(config_path)
    try:
        raw_config = json.loads(config_path.read_text())
    except OSError as error:
        raise ConfigError(f"cannot read the file ({error.strerror})") from error
    except json.JSONDecodeError as error:
        raise ConfigError(f"invalid JSON ({error})") from error
    if not isinstance(raw_config, dict):
        raise ConfigError("the top level of the JSON must be an object")

    # Field names and types (unknown names are rejected in both modes, to catch typos and outdated fields).
    known_fields = {field.name for field in fields(MixNMatchConfig)}
    required_fields = set(STANDARD_RUN_FIELDS) if standard_run else known_fields - {"crops"}
    unknown_fields = sorted(raw_config.keys() - known_fields)
    missing_fields = sorted(required_fields - raw_config.keys())
    if unknown_fields:
        raise ConfigError(f"unknown field(s) {unknown_fields}")
    if missing_fields:
        raise ConfigError(f"missing field(s) {missing_fields}")
    for field_name, expected_type in SCALAR_FIELD_TYPES.items():
        if field_name not in required_fields:
            continue
        value = raw_config[field_name]
        # JSON booleans are Python ints, so they must be rejected explicitly for numeric fields.
        is_wrong_bool = isinstance(value, bool) and expected_type is not bool
        if is_wrong_bool or not isinstance(value, expected_type):
            raise ConfigError(f'"{field_name}" has the wrong type ({type(value).__name__})')
    config = MixNMatchConfig(**{field_name: raw_config.get(field_name) for field_name in known_fields})

    # Values used by every run.
    if not config.prefix or "/" in config.prefix or "\\" in config.prefix:
        raise ConfigError('"prefix" must be a non-empty name without path separators')
    if standard_run:
        return config

    # Values used only by tiled runs.
    if config.num_crops < 1 or config.tiles_per_crop < 1:
        raise ConfigError('"num_crops" and "tiles_per_crop" must be at least 1')
    if not 0 <= config.late_prompting < NUM_STEPS:
        raise ConfigError(f'"late_prompting" must be between 0 and {NUM_STEPS - 1}')
    check_prompt_grid("tile_prompts", config.tile_prompts, config.num_crops, config.tiles_per_crop)
    check_prompt_grid("tile_negative_prompts", config.tile_negative_prompts, config.num_crops, config.tiles_per_crop)
    if not config.visualize and not config.save_separate_tiles:
        raise ConfigError('nothing would be saved: set "visualize" and/or "save_separate_tiles" to true')

    # Crop layout: fixed crops from the config, or chosen from the saliency measured at step late_prompting - 1.
    if config.static_cropping:
        if not isinstance(config.crops, list) or len(config.crops) != config.num_crops:
            raise ConfigError(f'"crops" must be a list of {config.num_crops} crops when "static_cropping" is true')
        build_crop_map(config.crops, IMAGE_WIDTH, IMAGE_HEIGHT)
    else:
        if config.late_prompting < 1:
            raise ConfigError('"late_prompting" must be at least 1 when "static_cropping" is false (the saliency is '
                              'measured at step late_prompting - 1)')
        if any(not prompt.strip() for crop_prompts in config.tile_prompts for prompt in crop_prompts):
            raise ConfigError('every tile prompt needs text when "static_cropping" is false (the saliency of an '
                              'empty prompt has no words)')
    return config
