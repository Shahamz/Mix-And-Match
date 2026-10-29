"""
Baseline generation: one plain PixelDiT image per tile prompt.

    python -m baselines.mnm_baseline configs/template.json
    python -m baselines.mnm_baseline configs/ --weights-dir <folder>

Nothing here splits, crops, tiles or masks: every tile prompt of a config is denoised on its own, at the full
image size, exactly as the unmodified model would. It is the control a Mix-And-Match run is read against - what a
prompt looks like when it has the whole image to itself and no other prompt to share attention with.

The background prompt takes no part. It is what a Mix-And-Match run denoises before the split, not one of the
prompts being compared, so a config of num_crops crops of tiles_per_crop tiles yields exactly
num_crops * tiles_per_crop images and no more.

Every image starts from the same noise, the config's seed being reused for each, so two baseline images differ by
their prompt alone. The images are produced one at a time rather than as a batch: a batch of them is the one thing
that would need more memory than the run it is a baseline for.

Sampling is the same as pipeline_mix_n_match's, because a baseline that sampled differently would not be one, so
the pipeline below inherits it: DPM-Solver++ (order 2) on flow sigmas with the checkpoint's flow shift,
classifier-free guidance from the tile's own negative prompt, bfloat16 model and float32 sample.

The images are written in the tiles layout of a real run, so they sit beside one and compare directly:
outputs/mnm_baseline/[<config folder>/]<prefix>_<num_crops>crops_<tiles_per_crop>tiles/tiles/crop<i>/tile<j>.png.
They are whole images, not cut tiles, so no layout.json is written: there is no crop map behind them.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers.utils.torch_utils import randn_tensor
from torchvision.transforms.functional import to_pil_image

from config import ConfigError, load_config
from macros import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_STEPS, OUTPUT_DIR, save_image
from pipeline_mix_n_match import MODEL_DTYPE, MixNMatchPipeline, encode_all_prompts, to_display_image

# Device the model runs on.
DEVICE = "cuda"

# Folder of this baseline's results inside outputs/.
BASELINE_DIRNAME = "mnm_baseline"


@dataclass
class BaselineOutput:
    """The images of one baseline run."""

    images: torch.Tensor  # [num_crops * tiles_per_crop, 3, H, W] in [0, 1], prompt crop * k + tile at that index


class BaselinePipeline(MixNMatchPipeline):
    """Denoises one full-size image per tile prompt, with no crops, tiles or masks of any kind."""

    def denoise(self, config, text_embeds):
        """
        Denoises one image from the config's seed with one prompt, as the unmodified model would.

        Args:
            config: MixNMatchConfig.
            text_embeds: [2, 1, PROMPT_LENGTH, text dim] one prompt's (negative, positive) features.

        Returns:
            [3, H, W] image in [0, 1].
        """
        device = self.device
        generator = torch.Generator(device=device).manual_seed(config.seed)
        sample = randn_tensor((1, 3, IMAGE_HEIGHT, IMAGE_WIDTH), generator=generator, device=device)
        self.scheduler.set_timesteps(NUM_STEPS, device=device)
        for step_index, timestep in enumerate(self.progress_bar(self.scheduler.timesteps)):
            sigma = self.scheduler.sigmas[step_index].item()
            velocity = self.predict_velocity(sample, sigma, text_embeds, None, config)
            sample = self.scheduler.step(velocity, timestep, sample).prev_sample
        return to_display_image(sample[0])

    @torch.no_grad()
    def __call__(self, config, encoded_prompts):
        """
        Runs one standard denoising per tile prompt.

        Args:
            config: MixNMatchConfig.
            encoded_prompts: this config's EncodedPrompts from encode_all_prompts.

        Returns:
            BaselineOutput.
        """
        text_embeds = encoded_prompts.embeddings.to(device=self.device, dtype=MODEL_DTYPE)
        images = []
        for prompt_index in range(config.num_crops * config.tiles_per_crop):
            crop_index, tile_index = divmod(prompt_index, config.tiles_per_crop)
            print(f"  crop {crop_index}, tile {tile_index}: {config.tile_prompts[crop_index][tile_index]}")
            # Prompt 0 is the background prompt, which a baseline ignores; the tile prompts follow it in order.
            images.append(self.denoise(config, text_embeds[:, prompt_index + 1:prompt_index + 2]))
        return BaselineOutput(images=torch.stack(images))


def save_baseline_run(output, config, output_root):
    """
    Saves a baseline run in the tiles layout, so its images sit beside a real run's and compare directly.

    Args:
        output: BaselineOutput.
        config: MixNMatchConfig.
        output_root: folder to write into: OUTPUT_DIR/mnm_baseline, plus the config folder's name for a folder run.

    Returns:
        Path of the run's tiles folder.
    """
    run_folder = output_root / f"{config.prefix}_{config.num_crops}crops_{config.tiles_per_crop}tiles"
    tiles_folder = run_folder / "tiles"
    for prompt_index, image in enumerate(output.images):
        crop_index, tile_index = divmod(prompt_index, config.tiles_per_crop)
        crop_folder = tiles_folder / f"crop{crop_index}"
        crop_folder.mkdir(parents=True, exist_ok=True)
        save_image(to_pil_image(image), crop_folder / f"tile{tile_index}.png")
    print(f"Saved baseline images: {tiles_folder}")
    return tiles_folder


def main():
    """Parses the arguments, runs the configs, and reports the results."""
    parser = argparse.ArgumentParser(description="Mix-And-Match baseline: one plain image per tile prompt.")
    parser.add_argument("config_path", type=Path, help="a config .json file, or a folder of them")
    parser.add_argument("--weights-dir", type=Path, default=None,
                        help="folder with config.json, pixeldit_t2i_v1.pth and gemma-2-2b-it/ (default: download)")
    args = parser.parse_args()

    # A folder run keeps its results together in outputs/mnm_baseline/<folder name>/.
    output_root = OUTPUT_DIR / BASELINE_DIRNAME
    if args.config_path.is_dir():
        config_paths = sorted(args.config_path.glob("*.json"))
        output_root = output_root / args.config_path.name
    elif args.config_path.is_file():
        config_paths = [args.config_path]
    else:
        sys.exit(f"Error: {args.config_path} does not exist")
    if not config_paths:
        sys.exit(f"Error: no .json files in {args.config_path}")
    if args.weights_dir is not None and not args.weights_dir.is_dir():
        sys.exit(f"Error: the weights folder {args.weights_dir} does not exist")
    if not torch.cuda.is_available():
        sys.exit("Error: the baseline needs a CUDA GPU")

    configs, failures = {}, {}
    for config_path in config_paths:
        try:
            configs[config_path] = load_config(config_path)
        except ConfigError as error:
            failures[config_path] = f"invalid config: {error}"

    if configs:
        try:
            all_encoded_prompts = encode_all_prompts(list(configs.values()), DEVICE, args.weights_dir)
            pipeline = BaselinePipeline.from_weights(DEVICE, args.weights_dir)
        except OSError as error:  # a missing file in --weights-dir, or a failed download
            sys.exit(f"Error: could not load the weights ({error})")
        for (config_path, config), encoded_prompts in zip(configs.items(), all_encoded_prompts, strict=True):
            print(f"\n=== {config_path.name} ===")
            try:
                save_baseline_run(pipeline(config, encoded_prompts), config, output_root)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                failures[config_path] = 'out of GPU memory (try "sequential_cfg": true)'
            except OSError as error:
                failures[config_path] = f"could not save the outputs ({error})"

    print(f"\nFinished: {len(config_paths) - len(failures)} of {len(config_paths)} config(s) succeeded.")
    for config_path, reason in failures.items():
        print(f"  FAILED {config_path}: {reason}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
