"""
Mix-And-Match pipeline for PixelDiT-T2I.

The run has two phases, so that the Gemma text encoder and the transformer never share the GPU:

    1. encode_all_prompts(configs): encodes every prompt of every config. Encoded prompts are cached on disk
       (PROMPT_CACHE_DIR), so Gemma is loaded only when some prompt was never encoded before.
    2. MixNMatchPipeline(config, encoded_prompts): the denoising loop, run once per config.

Denoising loop:
    - One noisy image G is denoised with the background prompt only (the original, unmasked model).
    - With dynamic cropping ("static_cropping": false), step late_prompting - 1 also measures where every prompt
      attends (saliency.py); the step's result is unchanged. The saliency maps give the crop map, including the
      background crop, which is not split into tiles.
    - Before step late_prompting, the crop map is taken from the saliency (dynamic) or from the config's crops
      (static). G is copied k times: image s holds tile s of every crop, so every tile starts equal to G. From then
      on the k images run as ONE sequence with the tile prompts and the tile attention rules (attention.py).
    - The result is k images in [0, 1]; tile (crop i, tile j) is the crop-i region of image j.
    - A standard run (standard_run=True, generate.py --std) never splits: plain PixelDiT with the background prompt.

Sampling matches PixelDiT's inference: DPM-Solver++ (order 2) on flow sigmas with the checkpoint's flow shift,
classifier-free guidance with one negative prompt per positive prompt, bfloat16 model, float32 sample. With
sequential_cfg, the negative and positive passes run one after the other instead of as one batch (same result,
about half the memory).
"""

import gc
import hashlib
from dataclasses import dataclass, replace

import numpy as np
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, Gemma2Model

from attention import build_tile_attention_layout
from config import build_crop_map
from macros import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_STEPS, PATCH_SIZE_PIXELS, PROJECT_DIR, PROMPT_LENGTH
from pixeldit_model import load_pixeldit
from saliency import SaliencyProbe, build_saliency_layout, crop_map_from_saliency, saliency_crop_maps

# The Gemma-2-2b-it copy PixelDiT was trained with, its folder name inside --weights-dir, and the files needed.
TEXT_ENCODER_REPO_ID = "Efficient-Large-Model/gemma-2-2b-it"
TEXT_ENCODER_DIRNAME = "gemma-2-2b-it"
TEXT_ENCODER_FILE_PATTERNS = ["*.json", "model-*.safetensors", "tokenizer.model"]

# Encoded prompts, one file per prompt; the folder can be deleted at any time (prompts are then re-encoded).
PROMPT_CACHE_DIR = PROJECT_DIR / "prompt_cache"

# Prompts encoded by Gemma at once.
ENCODE_BATCH_SIZE = 16

# "Complex human instruction" prepended to every positive prompt, copied from PixelDiT's stage-3 config.
COMPLEX_HUMAN_INSTRUCTION = "\n".join([
    'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:',
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
    "User Prompt: ",
])

# dtype of both models (PixelDiT's inference runs in bfloat16).
MODEL_DTYPE = torch.bfloat16

# The model's time input is sigma * NUM_TRAIN_TIMESTEPS.
NUM_TRAIN_TIMESTEPS = 1000

# Order of the DPM-Solver++ multistep sampler used by PixelDiT's inference.
SOLVER_ORDER = 2


def encode_prompts(prompts, is_positive, tokenizer, text_encoder, instruction_length):
    """
    Encodes a list of prompts the way PixelDiT's inference does.

    Args:
        prompts: list of strings.
        is_positive: True for positive prompts (instruction prefix, first + last PROMPT_LENGTH - 1 tokens kept),
            False for negative prompts (plain, PROMPT_LENGTH tokens).
        tokenizer: Gemma tokenizer (right padding).
        text_encoder: Gemma2Model.
        instruction_length: number of tokens of COMPLEX_HUMAN_INSTRUCTION, including the start token.

    Returns:
        (features, word_masks): [len(prompts), PROMPT_LENGTH, text dim] CPU tensor, and [len(prompts),
        PROMPT_LENGTH] bool CPU tensor marking the prompt's own words (no start token, instruction or padding).
    """
    if is_positive:
        prompts = [COMPLEX_HUMAN_INSTRUCTION + prompt for prompt in prompts]
    max_length = instruction_length + PROMPT_LENGTH - 2 if is_positive else PROMPT_LENGTH
    tokens = tokenizer(prompts, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    tokens = tokens.to(text_encoder.device)
    features = text_encoder(tokens.input_ids, attention_mask=tokens.attention_mask).last_hidden_state
    # The prompt's first word starts after the start token and, for positive prompts, after the instruction. The
    # instruction's trailing space merges into that first word's token, so it is counted without it.
    first_word_index = len(tokenizer.encode(COMPLEX_HUMAN_INSTRUCTION.rstrip())) if is_positive else 1
    word_masks = tokens.attention_mask.bool()
    word_masks[:, :first_word_index] = False
    if is_positive:
        kept_positive_tokens = [0] + list(range(-PROMPT_LENGTH + 1, 0))
        features, word_masks = features[:, kept_positive_tokens], word_masks[:, kept_positive_tokens]
    return features.cpu(), word_masks.cpu()


def prompt_cache_path(prompt, is_positive):
    """
    Args:
        prompt: prompt string.
        is_positive: whether it is encoded as a positive prompt (with the instruction).

    Returns:
        Path of the prompt's cache file, named by a hash of everything its encoding depends on.
    """
    kind = COMPLEX_HUMAN_INSTRUCTION if is_positive else "negative prompt"
    cache_key = "\n".join([TEXT_ENCODER_REPO_ID, str(PROMPT_LENGTH), kind, prompt])
    return PROMPT_CACHE_DIR / f"{hashlib.sha256(cache_key.encode()).hexdigest()}.pt"


@dataclass
class EncodedPrompts:
    """The encoded prompts of one config."""

    embeddings: torch.Tensor  # [2, prompts, PROMPT_LENGTH, text dim]: index 0 negative prompts, index 1 positive
    word_masks: torch.Tensor  # [prompts, PROMPT_LENGTH] bool: the words of every positive prompt, background first


@torch.no_grad()
def encode_all_prompts(configs, device, weights_dir=None, standard_run=False):
    """
    Encodes the prompts of all configs, from the cache where possible; Gemma is loaded once if any are missing.

    Processing follows PixelDiT's inference: positive prompts get the complex human instruction prefix and keep
    the first token plus the last PROMPT_LENGTH - 1 tokens; negative prompts are encoded plainly to PROMPT_LENGTH
    tokens. Padding tokens are kept (the model was trained without a text mask).

    Args:
        configs: list of MixNMatchConfig.
        device: torch device to run Gemma on.
        weights_dir: Path of a folder holding Gemma in gemma-2-2b-it/, or None to download it (Hugging Face cache).
        standard_run: True to encode only the background prompts (generate.py --std).

    Returns:
        List with one EncodedPrompts per config. Prompt 0 is the background prompt, prompt 1 + crop * k + tile is
        that tile's prompt (no tile prompts in a standard run).

    Raises:
        FileNotFoundError: if weights_dir has no gemma-2-2b-it/ folder.
    """
    config_prompts = []  # per config: (negative prompts, positive prompts)
    for config in configs:
        negative_prompts, positive_prompts = [config.background_negative_prompt], [config.background_prompt]
        if not standard_run:
            negative_prompts += [prompt for crop in config.tile_negative_prompts for prompt in crop]
            positive_prompts += [prompt for crop in config.tile_prompts for prompt in crop]
        config_prompts.append((negative_prompts, positive_prompts))

    # Keys are (prompt, is_positive); values are {"features", "word_mask"} dicts.
    unique_keys = {(prompt, is_positive) for prompt_lists in config_prompts
                   for is_positive, prompts in enumerate(prompt_lists) for prompt in prompts}
    encoded = {key: torch.load(prompt_cache_path(*key)) for key in unique_keys if prompt_cache_path(*key).exists()}
    missing_keys = sorted(unique_keys - encoded.keys())
    print(f"Prompt cache: {len(encoded)} of {len(unique_keys)} prompts cached")

    if missing_keys:
        if weights_dir is None:
            text_encoder_dir = snapshot_download(TEXT_ENCODER_REPO_ID, allow_patterns=TEXT_ENCODER_FILE_PATTERNS)
        else:
            text_encoder_dir = weights_dir / TEXT_ENCODER_DIRNAME
            if not text_encoder_dir.is_dir():
                raise FileNotFoundError(f"{text_encoder_dir} does not exist (expected {TEXT_ENCODER_REPO_ID})")
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_dir)
        tokenizer.padding_side = "right"
        text_encoder = Gemma2Model.from_pretrained(text_encoder_dir, dtype=MODEL_DTYPE).to(device).eval()
        instruction_length = len(tokenizer.encode(COMPLEX_HUMAN_INSTRUCTION))
        PROMPT_CACHE_DIR.mkdir(exist_ok=True)
        for is_positive in (False, True):
            prompts = [prompt for prompt, key_is_positive in missing_keys if key_is_positive == is_positive]
            for start in range(0, len(prompts), ENCODE_BATCH_SIZE):
                batch = prompts[start:start + ENCODE_BATCH_SIZE]
                features, word_masks = encode_prompts(batch, is_positive, tokenizer, text_encoder, instruction_length)
                for prompt, prompt_features, word_mask in zip(batch, features, word_masks, strict=True):
                    entry = {"features": prompt_features.clone(), "word_mask": word_mask.clone()}
                    torch.save(entry, prompt_cache_path(prompt, is_positive))
                    encoded[(prompt, is_positive)] = entry
        del text_encoder
        gc.collect()
        torch.cuda.empty_cache()

    return [
        EncodedPrompts(
            embeddings=torch.stack([
                torch.stack([encoded[(prompt, False)]["features"] for prompt in negative_prompts]),
                torch.stack([encoded[(prompt, True)]["features"] for prompt in positive_prompts]),
            ]),
            word_masks=torch.stack([encoded[(prompt, True)]["word_mask"] for prompt in positive_prompts]),
        )
        for negative_prompts, positive_prompts in config_prompts
    ]


def to_display_image(image):
    """
    Converts a model-space image to a displayable one.

    Args:
        image: [3, H, W] tensor, nominally in [-1, 1].

    Returns:
        [3, H, W] float CPU tensor clamped to [0, 1].
    """
    return ((image.float().clamp(-1, 1) + 1) / 2).cpu()


@dataclass
class MixNMatchOutput:
    """Result of one Mix-And-Match run."""

    tiles: torch.Tensor  # [k, 3, H, W] in [0, 1]; image s holds tile s of every crop ([1, 3, H, W] in a standard run)
    crop_map: np.ndarray | None  # [patch rows, patch cols] crop index of every patch; None in a standard run
    saliency_maps: torch.Tensor | None = None  # [1 + tile prompts, patch rows, patch cols]; None with static crops
    saliency_crop_maps: np.ndarray | None = None  # [num_crops + 1, patch rows, patch cols], background last
    saliency_image: torch.Tensor | None = None  # [3, H, W] x0 prediction of the saliency step, in [0, 1]


class MixNMatchPipeline(DiffusionPipeline):
    """Denoises one image with a background prompt, then splits it into k tiles per crop with their own prompts."""

    def __init__(self, transformer, scheduler):
        """
        Args:
            transformer: PixDiT_T2I model.
            scheduler: DPMSolverMultistepScheduler configured for flow matching, with the checkpoint's flow shift.
        """
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler)

    @classmethod
    def from_weights(cls, device, weights_dir=None):
        """
        Loads the PixelDiT transformer (see load_pixeldit) and builds PixelDiT's sampler.

        Args:
            device: torch device for the transformer.
            weights_dir: Path of a folder holding the checkpoint, or None to download it (Hugging Face cache).

        Returns:
            MixNMatchPipeline.
        """
        transformer, model_config = load_pixeldit(device, MODEL_DTYPE, weights_dir)
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=NUM_TRAIN_TIMESTEPS,
            solver_order=SOLVER_ORDER,
            algorithm_type="dpmsolver++",
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            flow_shift=model_config["scheduler"]["flow_shift"],
        )
        return cls(transformer, scheduler)

    def predict_velocity(self, sample, sigma, text_embeds, layout, config):
        """
        Runs the transformer for both classifier-free guidance passes and combines them.

        Args:
            sample: [S, 3, H, W] float32 current images.
            sigma: current noise level.
            text_embeds: [2, prompts, PROMPT_LENGTH, text dim] (negative, positive).
            layout: TileAttentionLayout or None.
            config: MixNMatchConfig (cfg_scale, sequential_cfg).

        Returns:
            [S, 3, H, W] float32 guided velocity.
        """
        images = sample.unsqueeze(0).to(MODEL_DTYPE)
        # The model gets the exact time sigma * 1000 (scheduler.timesteps are rounded), in the model dtype.
        model_time = torch.full((1,), sigma * NUM_TRAIN_TIMESTEPS, device=sample.device, dtype=MODEL_DTYPE)
        if config.sequential_cfg:
            # The saliency probe reads the last batch element, so it may only see the positive pass.
            negative_layout = None if layout is None else replace(layout, saliency_probe=None)
            negative_velocity = self.transformer(images, model_time, text_embeds[:1], negative_layout)[0].float()
            positive_velocity = self.transformer(images, model_time, text_embeds[1:], layout)[0].float()
        else:
            cfg_images, cfg_time = images.expand(2, -1, -1, -1, -1), model_time.expand(2)
            negative_velocity, positive_velocity = self.transformer(cfg_images, cfg_time, text_embeds, layout).float()
        return negative_velocity + config.cfg_scale * (positive_velocity - negative_velocity)

    @torch.no_grad()
    def __call__(self, config, encoded_prompts, standard_run=False):
        """
        Runs one config.

        Args:
            config: MixNMatchConfig.
            encoded_prompts: this config's EncodedPrompts from encode_all_prompts.
            standard_run: True to never split, i.e. standard PixelDiT denoising with the background prompt only.

        Returns:
            MixNMatchOutput.
        """
        device = self.device
        patch_rows, patch_cols = IMAGE_HEIGHT // PATCH_SIZE_PIXELS, IMAGE_WIDTH // PATCH_SIZE_PIXELS
        text_embeds = encoded_prompts.embeddings.to(device=device, dtype=MODEL_DTYPE)
        generator = torch.Generator(device=device).manual_seed(config.seed)
        sample = randn_tensor((1, 3, IMAGE_HEIGHT, IMAGE_WIDTH), generator=generator, device=device)
        self.scheduler.set_timesteps(NUM_STEPS, device=device)
        saliency_step = None if standard_run or config.static_cropping else config.late_prompting - 1

        output = MixNMatchOutput(tiles=sample, crop_map=None)
        layout = None
        for step_index, timestep in enumerate(self.progress_bar(self.scheduler.timesteps)):
            if not standard_run and step_index == config.late_prompting:
                if config.static_cropping:
                    output.crop_map = build_crop_map(config.crops, IMAGE_WIDTH, IMAGE_HEIGHT)
                else:
                    empty_crops = sorted(set(range(config.num_crops)) - set(np.unique(output.crop_map).tolist()))
                    if empty_crops:
                        print(f"Warning: crop(s) {empty_crops} won no patch in the saliency maps; continuing "
                              "without them")
                # Split: copy G k times. The scheduler's stored model outputs (scheduler internals) are copied the
                # same way, so the multistep solver continues exactly.
                num_tiles = config.tiles_per_crop
                self.scheduler.model_outputs = [
                    None if model_output is None else model_output.repeat(num_tiles, 1, 1, 1)
                    for model_output in self.scheduler.model_outputs
                ]
                layout = build_tile_attention_layout(output.crop_map, config.num_crops, num_tiles, device)
                sample = sample.repeat(num_tiles, 1, 1, 1)

            # Before the split only the background prompt (prompt 0) is used; the saliency step adds the tile
            # prompts, which the image never attends (see saliency.py).
            step_layout = layout
            step_text = text_embeds if layout is not None else text_embeds[:, :1]
            if step_index == saliency_step:
                probe = SaliencyProbe(encoded_prompts.word_masks.to(device))
                step_layout = build_saliency_layout(len(encoded_prompts.word_masks) - 1, patch_rows * patch_cols,
                                                    probe, device)
                step_text = text_embeds
            sigma = self.scheduler.sigmas[step_index].item()
            velocity = self.predict_velocity(sample, sigma, step_text, step_layout, config)

            if step_index == saliency_step:
                predicted_clean = sample - sigma * velocity  # flow matching: velocity = noise - clean image
                output.saliency_maps, output.saliency_crop_maps = saliency_crop_maps(
                    probe, patch_rows, patch_cols, config.tiles_per_crop
                )
                # The crop map of the split: the tile crops, plus the background crop (index num_crops).
                output.crop_map = crop_map_from_saliency(output.saliency_crop_maps)
                output.saliency_image = to_display_image(predicted_clean[0])
            sample = self.scheduler.step(velocity, timestep, sample).prev_sample

        output.tiles = torch.stack([to_display_image(image) for image in sample])
        return output
