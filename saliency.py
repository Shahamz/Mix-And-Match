"""
Saliency of the prompts over the image, and the crop layout it suggests (dynamic cropping).

When: at step late_prompting - 1 of a run with "static_cropping": false. There is one image and no crops or tiles
yet.

How: the tile prompts are added to that step's forward pass. The attention mask keeps the real computation
unchanged: the background prompt and the image attend only [background prompt | image], so the tile prompts never
change the image; each tile prompt attends [its own tokens | image]. In every patch-stage block, SaliencyProbe reads
the queries and keys the model computed (after norm and RoPE) and measures, on the side, "which prompt does this
patch look at?": every patch query takes a softmax over the words of ALL prompts (the background prompt and every
tile prompt), each prompt keeps the max over its words, and per patch the prompts' values are renormalized to sum
to 1. "Words" are a prompt's own tokens (no instruction, no padding). The maps are averaged over heads and blocks
and smoothed with a Gaussian of GAUSSIAN_SIGMA_PIXELS.

Crop maps: one map per crop, the mean over the crop's tile prompts, followed by the background crop (LAST), whose
map is the background prompt's map. Every crop map is then stretched to [0, 1] on its own (min-max), so only its
shape decides, not its overall strength.

Segmentation (which crop owns which patch): a Potts MRF over the patch grid,

    minimize  sum_p U(p, label of p) + MRF_SMOOTHNESS * (number of neighbouring patches with different labels),

with U(p, i) = 1 - S_i(p), solved by alpha-expansion (PyMaxflow). Every tile crop's PEAK (the patch where its map
is highest) is pinned to it, so no tile crop vanishes; the background crop is never pinned, so it keeps only the
patches it wins. The region it wins is not split into tiles; attention.py keeps its patches once and lets only the
background prompt's side see them.
"""

import math

import numpy as np
import torch
from maxflow.fastmin import aexpansion_grid
from PIL import Image, ImageOps
from torchvision.transforms.functional import gaussian_blur, to_pil_image

from attention import (BACKGROUND_PROMPT_TOKEN, NO_INDEX, TILE_PROMPT_TOKEN, TILE_TOKEN, TileAttentionLayout,
                       compiled_create_block_mask)
from macros import PATCH_SIZE_PIXELS, PROMPT_LENGTH, save_image

# Standard deviation, in pixels, of the Gaussian that smooths the saliency maps.
GAUSSIAN_SIGMA_PIXELS = 32

# Weight of a differing neighbour pair in the MRF energy.
MRF_SMOOTHNESS = 0.02

# Unary cost that pins a peak patch to its crop (far above any real cost, which is at most 1).
PINNED_COST = 1e6

# Highest number of alpha-expansion cycles.
MAX_MRF_PASSES = 10

# Heat map colors from the lowest to the highest value of a map (PIL ImageOps.colorize).
HEAT_COLORS = {"black": "blue", "mid": "yellow", "white": "red"}

# Weight of the heat map or segmentation colors when blended over the x0 prediction.
OVERLAY_OPACITY = 0.5

# Colors of the crops in the segmentation image (cycled when there are more crops).
SEGMENTATION_COLORS = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
]


def build_saliency_layout(num_tile_prompts, num_patches, probe, device):
    """
    Builds the attention layout of the saliency step: sequence [background prompt | tile prompts | image].

    Background prompt and image queries attend [background prompt | image] (the real, unchanged computation); a
    tile prompt's queries attend its own tokens and the image. The pixel stage keeps full attention.

    Args:
        num_tile_prompts: number of tile prompts (num_crops * tiles_per_crop).
        num_patches: patches of the image.
        probe: SaliencyProbe that records the prompt-image attention.
        device: torch device for the mask.

    Returns:
        TileAttentionLayout carrying the probe.
    """
    prompt_indices = torch.arange(1 + num_tile_prompts, device=device).repeat_interleave(PROMPT_LENGTH)
    kinds = torch.cat([
        torch.where(prompt_indices > 0, TILE_PROMPT_TOKEN, BACKGROUND_PROMPT_TOKEN),
        torch.full((num_patches,), TILE_TOKEN, device=device),
    ])
    owners = torch.cat([prompt_indices, torch.full((num_patches,), NO_INDEX, device=device)])

    def mask_mod(batch, head, query_index, key_index):
        query_kind, key_kind = kinds[query_index], kinds[key_index]
        key_is_image = key_kind == TILE_TOKEN
        real_query_allowed = (query_kind != TILE_PROMPT_TOKEN) & (key_is_image | (key_kind == BACKGROUND_PROMPT_TOKEN))
        tile_prompt_query_allowed = (query_kind == TILE_PROMPT_TOKEN) & (
            key_is_image | (owners[query_index] == owners[key_index])
        )
        return real_query_allowed | tile_prompt_query_allowed

    length = len(kinds)
    block_mask = compiled_create_block_mask(mask_mod, None, None, length, length, device=device)
    # One image, no crops yet: every patch is a "tile patch", and there are no crop averages or background patches.
    return TileAttentionLayout(num_tiles=1, append_crop_averages=False, joint_block_mask=block_mask,
                               pixel_block_mask=None, tile_patch_count=num_patches, saliency_probe=probe)


class SaliencyProbe:
    """Measures patch-to-prompt attention in every patch-stage block of one forward pass (see the top of this file)."""

    def __init__(self, word_masks):
        """
        Args:
            word_masks: [1 + tile prompts, PROMPT_LENGTH] bool tensor marking every prompt's words; row 0 is the
                background prompt.
        """
        self.num_prompts = len(word_masks)
        # Text position and owning prompt of every word.
        owners, word_indices = torch.nonzero(word_masks, as_tuple=True)
        self.word_positions = owners * PROMPT_LENGTH + word_indices
        self.word_owners = owners
        self.map_sum = 0  # sum over blocks of the [prompts, patches] maps
        self.num_blocks = 0

    def record(self, image_queries, text_keys):
        """
        Adds one block's maps. The last batch element is the positive CFG pass.

        Args:
            image_queries: [batch, patches, heads, head dim] after norm and RoPE.
            text_keys: [batch, text length, heads, head dim] after norm and RoPE.
        """
        scale = image_queries.shape[-1] ** -0.5
        word_keys = text_keys[-1, self.word_positions].float()
        # [heads, patches, words]: every patch's softmax over the words of all prompts.
        probabilities = torch.einsum("phd,whd->hpw", image_queries[-1].float(), word_keys).mul(scale).softmax(-1)

        # Every prompt keeps the max over its words: the word dimension becomes a prompt dimension.
        owner_index = self.word_owners.view(1, 1, -1).expand_as(probabilities)
        prompt_shape = [*probabilities.shape[:2], self.num_prompts]
        prompt_maps = torch.zeros(prompt_shape, device=probabilities.device).scatter_reduce(
            2, owner_index, probabilities, "amax", include_self=False
        )
        prompt_maps = prompt_maps / prompt_maps.sum(dim=2, keepdim=True)  # per patch across the prompts
        prompt_maps = prompt_maps.mean(dim=0)  # over heads
        self.map_sum = self.map_sum + prompt_maps.T
        self.num_blocks += 1


def saliency_crop_maps(probe, patch_rows, patch_cols, num_tiles):
    """
    Turns the probe's measurements into smoothed prompt maps and normalized crop maps.

    Args:
        probe: SaliencyProbe after the forward pass.
        patch_rows, patch_cols: patch grid of the image.
        num_tiles: k, tiles per crop.

    Returns:
        (prompt_maps, crop_maps):
            prompt_maps: [1 + tile prompts, patch_rows, patch_cols] float CPU tensor, the background prompt first;
                the tile prompts follow in config order (prompt 1 + i * k + j).
            crop_maps: [num_crops + 1, patch_rows, patch_cols] numpy array in [0, 1]; the background crop is last.
    """
    # Average over blocks and smooth with a Gaussian that covers +-3 sigma.
    maps = (probe.map_sum / probe.num_blocks).view(probe.num_prompts, 1, patch_rows, patch_cols)
    sigma_patches = GAUSSIAN_SIGMA_PIXELS / PATCH_SIZE_PIXELS
    kernel_size = 2 * math.ceil(3 * sigma_patches) + 1
    prompt_maps = gaussian_blur(maps, [kernel_size, kernel_size], [sigma_patches, sigma_patches])[:, 0].cpu()

    # One map per crop: the mean over its tile prompts; the background prompt's map becomes the last crop.
    background, tile_maps = prompt_maps[:1], prompt_maps[1:]
    crop_maps = tile_maps.reshape(-1, num_tiles, patch_rows, patch_cols).mean(dim=1)
    crop_maps = torch.cat([crop_maps, background]).numpy()

    # Stretch every crop map to [0, 1] on its own.
    highest = crop_maps.max(axis=(1, 2), keepdims=True)
    lowest = crop_maps.min(axis=(1, 2), keepdims=True)
    crop_maps = (crop_maps - lowest) / np.maximum(highest - lowest, 1e-12)
    return prompt_maps, crop_maps


def crop_peaks(crop_maps):
    """
    Finds every crop's peak patch: where its map is highest. Two crops cannot share a peak; the crop with the
    lower value there takes its next best patch that is not a peak yet.

    Args:
        crop_maps: [crops, patch rows, patch cols] array.

    Returns:
        List with the (row, col) peak of every crop.
    """
    flat = crop_maps.reshape(len(crop_maps), -1)
    peaks = {}  # flat patch index -> crop that holds it
    for crop_index in np.argsort(-flat.max(axis=1)):  # strongest crop first, so it keeps its own peak
        for patch in np.argsort(-flat[crop_index]):
            if patch not in peaks:
                peaks[patch] = crop_index
                break
    patch_of_crop = {crop_index: patch for patch, crop_index in peaks.items()}
    return [divmod(int(patch_of_crop[crop_index]), crop_maps.shape[2]) for crop_index in range(len(crop_maps))]


def crop_map_from_saliency(crop_maps):
    """
    Builds the crop layout from the crop maps with the MRF of the top of this file (alpha-expansion).

    Args:
        crop_maps: [num_crops + 1, patch rows, patch cols] array from saliency_crop_maps (background last).

    Returns:
        [patch rows, patch cols] numpy array of crop indices; num_crops marks the background crop.
    """
    # The tile crops' peaks are pinned; the background crop (last) is not.
    peaks = crop_peaks(crop_maps[:-1])
    segmentation = crop_maps.argmax(axis=0)
    for crop_index, (row, col) in enumerate(peaks):
        segmentation[row, col] = crop_index

    # Unary cost U(p, i) = 1 - S_i(p), with every peak forced to its own crop.
    unary = 1.0 - crop_maps
    for crop_index, (row, col) in enumerate(peaks):
        unary[:, row, col] = PINNED_COST
        unary[crop_index, row, col] = 0.0

    pairwise = MRF_SMOOTHNESS * (1.0 - np.eye(len(unary)))  # the Potts penalty, a metric as alpha-expansion requires
    # PyMaxflow wants the label axis last, and starts from (and writes into) the labels it is given.
    costs = np.moveaxis(unary, 0, -1).astype(np.float64)
    return aexpansion_grid(costs, pairwise, max_cycles=MAX_MRF_PASSES, labels=segmentation.astype(np.int8)).astype(int)


def save_heat_map(values, background, path, value_range=None):
    """
    Saves one map as a blue -> yellow -> red heat map over the image.

    Args:
        values: [patch rows, patch cols] numpy array.
        background: PIL image to blend with (the x0 prediction).
        path: Path of the .png to write.
        value_range: (lowest, highest) of the color scale. None scales this map on its own, which is fine for one
            map but makes two maps incomparable; the crop maps share one range so the reddest map at a patch is
            the crop that wins it.
    """
    lowest, highest = (values.min(), values.max()) if value_range is None else value_range
    scaled = (values - lowest) / max(highest - lowest, 1e-12)
    gray = Image.fromarray((scaled * 255).astype(np.uint8)).resize(background.size, Image.BILINEAR)
    save_image(Image.blend(background, ImageOps.colorize(gray, **HEAT_COLORS), OVERLAY_OPACITY), path)


def save_saliency_outputs(prompt_maps, crop_maps, segmentation, clean_image, num_tiles, folder):
    """
    Saves a heat map per prompt (c_bg.png, c<i>_t<j>.png) and per crop (crop<i>.png, crop_bg.png), plus the
    segmentation (crop_segmentation.png), all over the x0 prediction.

    Args:
        prompt_maps: [1 + tile prompts, patch rows, patch cols] saliency maps, the background prompt first (each
            drawn on its own scale: it shows where that prompt looks, not how it ranks against the others).
        crop_maps: [num_crops + 1, patch rows, patch cols] normalized maps, background last; they share one color
            scale, so the reddest crop map at a patch is the crop that wins it.
        segmentation: [patch rows, patch cols] crop indices from crop_map_from_saliency.
        clean_image: [3, H, W] x0 prediction of the saliency step, in [0, 1].
        num_tiles: k, tiles per crop.
        folder: Path to write to (created if needed).
    """
    folder.mkdir(parents=True, exist_ok=True)
    background = to_pil_image(clean_image)
    save_heat_map(prompt_maps[0].numpy(), background, folder / "c_bg.png")
    for prompt_index, prompt_map in enumerate(prompt_maps[1:]):
        crop_index, tile_index = divmod(prompt_index, num_tiles)
        save_heat_map(prompt_map.numpy(), background, folder / f"c{crop_index}_t{tile_index}.png")

    crop_range = (crop_maps.min(), crop_maps.max())
    crop_names = [f"crop{crop_index}.png" for crop_index in range(len(crop_maps) - 1)] + ["crop_bg.png"]
    for crop_map, name in zip(crop_maps, crop_names, strict=True):
        save_heat_map(crop_map, background, folder / name, crop_range)

    colors = np.array(SEGMENTATION_COLORS, dtype=np.uint8)[segmentation % len(SEGMENTATION_COLORS)]
    segmentation_image = Image.fromarray(colors).resize(background.size, Image.NEAREST)
    save_image(Image.blend(background, segmentation_image, OVERLAY_OPACITY), folder / "crop_segmentation.png")
