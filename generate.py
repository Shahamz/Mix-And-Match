"""
Command-line entry point of Mix-And-Match.

    python generate.py configs/template.json                 # one config
    python generate.py configs/template.json --vis_app       # then open the viewer on its tiles folder
    python generate.py configs/                              # every *.json in the folder, in name order;
                                                             # its results go to outputs/<folder name>/
    python generate.py configs/template.json --std           # standard denoising, background prompt only
    python generate.py configs/template.json --weights-dir W # read the weights from W instead of downloading them

All configs are validated first (invalid ones are reported and skipped). Then every prompt is taken from the prompt
cache or encoded by Gemma (loaded only if needed, then freed), the transformer is loaded once, and the configs run
one after another. A failing config is reported and the rest continue; failures are listed at the end.

Weights: by default PixelDiT and Gemma are downloaded once into the Hugging Face cache. With --weights-dir, the
folder must hold config.json and pixeldit_t2i_v1.pth (from nvidia/PixelDiT-1300M-1024px) and gemma-2-2b-it/ (from
Efficient-Large-Model/gemma-2-2b-it).

With --std, every config runs as plain PixelDiT with its background prompt only; tile and crop fields are ignored
and the image is saved as outputs/<prefix>_std_run.png.
"""

import argparse
import sys
from pathlib import Path

import torch

from config import ConfigError, load_config
from macros import OUTPUT_DIR
from pipeline_mix_n_match import MixNMatchPipeline, encode_all_prompts
from vis_app import run_viewer, save_outputs, save_standard_run

# Device the models run on.
DEVICE = "cuda"


def main():
    """Parses the arguments, runs the configs, and reports the results."""
    parser = argparse.ArgumentParser(description="Mix-And-Match: tiled prompting with PixelDiT.")
    parser.add_argument("config_path", type=Path, help="a config .json file, or a folder of them")
    parser.add_argument("--vis_app", action="store_true", help="open the viewer on the tiles folder when done")
    parser.add_argument("--std", action="store_true",
                        help="standard denoising with the background prompt only (tile and crop fields are ignored)")
    parser.add_argument("--weights-dir", type=Path, default=None,
                        help="folder with config.json, pixeldit_t2i_v1.pth and gemma-2-2b-it/ (default: download)")
    args = parser.parse_args()

    if args.std and args.vis_app:
        print("Warning: --vis_app is ignored with --std (its image is not made of tiles)")
        args.vis_app = False
    # A folder run keeps its results together in outputs/<folder name>/; a single config writes into outputs/.
    output_root = OUTPUT_DIR
    if args.config_path.is_dir():
        config_paths = sorted(args.config_path.glob("*.json"))
        output_root = OUTPUT_DIR / args.config_path.name
        if args.vis_app:
            print("Warning: --vis_app is ignored for folder runs; open a result with: python vis_app.py <tiles folder>")
            args.vis_app = False
    elif args.config_path.is_file():
        config_paths = [args.config_path]
    else:
        sys.exit(f"Error: {args.config_path} does not exist")
    if not config_paths:
        sys.exit(f"Error: no .json files in {args.config_path}")
    if args.weights_dir is not None and not args.weights_dir.is_dir():
        sys.exit(f"Error: the weights folder {args.weights_dir} does not exist")
    if not torch.cuda.is_available():
        sys.exit("Error: Mix-And-Match needs a CUDA GPU")

    configs, failures = {}, {}
    for config_path in config_paths:
        try:
            configs[config_path] = load_config(config_path, standard_run=args.std)
        except ConfigError as error:
            failures[config_path] = f"invalid config: {error}"
    if args.vis_app and configs and not next(iter(configs.values())).save_separate_tiles:
        sys.exit('Error: --vis_app needs "save_separate_tiles": true in the config')

    tiles_folder = None
    if configs:
        try:
            all_encoded_prompts = encode_all_prompts(list(configs.values()), DEVICE, args.weights_dir,
                                                     standard_run=args.std)
            pipeline = MixNMatchPipeline.from_weights(DEVICE, args.weights_dir)
        except OSError as error:  # a missing file in --weights-dir, or a failed download
            sys.exit(f"Error: could not load the weights ({error})")
        for (config_path, config), encoded_prompts in zip(configs.items(), all_encoded_prompts, strict=True):
            print(f"\n=== {config_path.name} ===")
            try:
                output = pipeline(config, encoded_prompts, standard_run=args.std)
                if args.std:
                    save_standard_run(output, config, output_root)
                else:
                    tiles_folder = save_outputs(output, config, output_root)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                failures[config_path] = ('out of GPU memory (try "sequential_cfg": true, or fewer crops or tiles '
                                         'per crop)')
            except OSError as error:
                failures[config_path] = f"could not save the outputs ({error})"

    print(f"\nFinished: {len(config_paths) - len(failures)} of {len(config_paths)} config(s) succeeded.")
    for config_path, reason in failures.items():
        print(f"  FAILED {config_path}: {reason}")
    if args.vis_app and tiles_folder is not None:
        run_viewer(tiles_folder)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
