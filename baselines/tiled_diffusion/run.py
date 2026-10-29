"""
Command line entry point of the tiled-diffusion baseline: generate candidate tiles for a config's crops.

    python -m baselines.tiled_diffusion.run configs/template.json
    python -m baselines.tiled_diffusion.run configs/template.json --no-combos --weights-dir <folder>

The crops are laid out in the same near-square grid mnm_baseline's images are assembled into, and every tile is
generated as one full size square, so the config's crop rectangles only decide the crops' order in the grid and a
combination is the tiles side by side in it (crop_layout.py).
The seed and the guidance scale come from the config, the step count is the project's, and the model is PixelDiT.

Results go to outputs/tiled_diffusion/<prefix>_<num_crops>crops_<tiles_per_crop>tiles/: tiles/, combos/ and
run_meta.json.

Use --dry-run to check the config and the layout without loading the model.
"""

import argparse
import gc
import sys
from pathlib import Path

import torch

from config import ConfigError, load_config
from macros import OUTPUT_DIR
from pipeline_mix_n_match import encode_all_prompts

from .crop_layout import build_latents, describe
from .crop_output import save_all_combinations, save_run_meta, save_tiles
from .model import PixelDiTLatentTiling

# Device the model runs on.
DEVICE = "cuda"

# Folder of this baseline's results inside outputs/.
BASELINE_DIRNAME = "tiled_diffusion"

# Seam context and similarity band, in 16 pixel patches. 4 patches is a 64 pixel context, which is enough for the
# seam without taking over the tile; the band is half of it, the nearest equivalent of the original's 5 latent
# pixels of Stable Diffusion.
DEFAULT_MAX_WIDTH = 4
DEFAULT_MAX_REPLICA_WIDTH = 2


def main():
    """Parses the arguments, builds the tiles, runs them, and writes the images."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config_path", type=Path, help="a config .json file")
    parser.add_argument("--weights-dir", type=Path, default=None,
                        help="folder with config.json, pixeldit_t2i_v1.pth and gemma-2-2b-it/ (default: download)")
    parser.add_argument("--max-width", type=int, default=DEFAULT_MAX_WIDTH,
                        help=f"context size used for the tiling constraint, in 16 pixel patches "
                             f"(default: {DEFAULT_MAX_WIDTH})")
    parser.add_argument("--max-replica-width", type=int, default=DEFAULT_MAX_REPLICA_WIDTH,
                        help=f"width of the similarity constraint band, in the same units "
                             f"(default: {DEFAULT_MAX_REPLICA_WIDTH})")
    parser.add_argument("--combos", action=argparse.BooleanOptionalAction, default=True,
                        help="save one full size image per combination of candidates, on top of the separate "
                             "tiles (default: --combos)")
    parser.add_argument("--max-combos", type=int, default=0,
                        help="cap on how many combinations to save; 0 means all (default: 0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the tile layout and exit, without loading the model")
    parser.add_argument("--show", action="store_true",
                        help="also display the first combination with matplotlib when finished (needs --combos)")
    args = parser.parse_args()

    if args.max_width < 1 or args.max_replica_width < 1:
        sys.exit("Error: --max-width and --max-replica-width count 16 pixel patches, so both must be at least 1")
    try:
        config = load_config(args.config_path)
    except ConfigError as error:
        sys.exit(f"Error: invalid config: {error}")
    if args.weights_dir is not None and not args.weights_dir.is_dir():
        sys.exit(f"Error: the weights folder {args.weights_dir} does not exist")

    latents_arr, placed_crops = build_latents(config)
    print(describe(config, latents_arr, placed_crops))
    if args.dry_run:
        print("--dry-run: stopping before loading the model.")
        return
    if not torch.cuda.is_available():
        sys.exit("Error: the baseline needs a CUDA GPU")

    # Only now do we pay for downloading / loading the models onto the GPU: the prompts first, so that Gemma is
    # gone before the transformer arrives.
    try:
        encoded_prompts = encode_all_prompts([config], DEVICE, args.weights_dir)[0]
    except OSError as error:
        sys.exit(f"Error: could not load the weights ({error})")
    torch.cuda.empty_cache()
    gc.collect()

    model = PixelDiTLatentTiling(args.weights_dir)
    try:
        latents_arr = model(latents_arr=latents_arr,
                            encoded_prompts=encoded_prompts,
                            config=config,
                            max_width=args.max_width,
                            max_replica_width=args.max_replica_width,
                            device=DEVICE)
    except OSError as error:
        sys.exit(f"Error: could not load the weights ({error})")
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        sys.exit("Error: out of GPU memory (try fewer crops or tiles per crop)")

    torch.cuda.empty_cache()
    gc.collect()

    out_dir = OUTPUT_DIR / BASELINE_DIRNAME / f"{config.prefix}_{config.num_crops}crops_{config.tiles_per_crop}tiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_tiles(latents_arr, placed_crops, out_dir)
    combo_paths = []
    if args.combos:
        combo_paths = save_all_combinations(latents_arr, placed_crops, out_dir, max_combos=args.max_combos)
    else:
        print("--no-combos: skipping the combination images, the tiles are in tiles/")
    save_run_meta(config, args, placed_crops, out_dir)

    if args.show and not combo_paths:
        print("--show needs a combination image, and none were saved.")
    elif args.show:
        import matplotlib.pyplot as plt
        from PIL import Image

        plt.imshow(Image.open(combo_paths[0]))
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    main()
