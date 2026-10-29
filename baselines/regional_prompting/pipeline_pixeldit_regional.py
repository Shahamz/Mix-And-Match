"""
Regional Prompting for PixelDiT: the pipeline, the regional attention mask, and its block-mask form.

The algorithm is unchanged from the FLUX original: every region gets its own prompt, the concatenated region
prompts and the image tokens are masked against each other in the joint attention, an unmasked "base" stream runs
alongside and is blended into the image tokens by base_ratio, and all of it only for the first mask_inject_steps
denoising steps, after which the model runs plainly on the background prompt. The mask is built by the same code
as in the original (pipeline_flux_regional.py:403-443).

What differs is everything model-shaped, and only because PixelDiT differs from FLUX: it works in pixel space with
16 px patches, so the token grid is height/16 x width/16 and there is no VAE; FLUX.1-dev is guidance-distilled and
needs no negative pass, while PixelDiT needs real classifier-free guidance, so the negative pass mirrors the
positive one (regional negative prompts, the same mask, its own base stream) and the two run one after the other
rather than as a batch of 2; and the sampler is PixelDiT's, DPM-Solver++ multistep over flow-matching sigmas with
the checkpoint's flow shift, fed the exact time sigma * 1000.

Prompts are encoded by the project's own encode_all_prompts, which is the same Gemma encoding, with the same
cache, that generate.py uses.
"""

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from torch.nn.attention.flex_attention import create_block_mask

from macros import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_STEPS, PATCH_SIZE_PIXELS, PROMPT_LENGTH
from pipeline_mix_n_match import MODEL_DTYPE, NUM_TRAIN_TIMESTEPS, SOLVER_ORDER, to_display_image
from pixeldit_model import load_pixeldit

from .transformer_regional import RegionalPixDiT_T2I


def build_regional_attention_mask(regional_masks, device):
    """
    Builds the regional attention mask over [region prompts | image patches].

    This is Regional-Prompting-FLUX's own construction (pipeline_flux_regional.py:388-443), unchanged apart from
    the token grid being PixelDiT's patch grid instead of FLUX's latent grid. True means the pair may attend.
    Every region's prompt sees only itself and its own patches; a patch sees the prompts of the regions it belongs
    to, patches sharing a region with it, and any patch that belongs to no region at all.

    Args:
        regional_masks: list of [IMAGE_HEIGHT, IMAGE_WIDTH] tensors, one per region, 1 inside the region.
        device: torch device the mask is built on.

    Returns:
        [regions * PROMPT_LENGTH + patches, same] boolean tensor.
    """
    H, W = IMAGE_HEIGHT // PATCH_SIZE_PIXELS, IMAGE_WIDTH // PATCH_SIZE_PIXELS
    hidden_seq_len = H * W

    masks = []
    for mask in regional_masks:
        # Resize regional masks to the token grid; the flatten is to match the sequence length.
        mask = torch.nn.functional.interpolate(mask[None, None, :, :].to(device), (H, W), mode='nearest-exact')
        masks.append(mask.flatten().unsqueeze(1).repeat(1, PROMPT_LENGTH))

    num_of_regions = len(masks)
    encoder_seq_len = num_of_regions * PROMPT_LENGTH
    each_prompt_seq_len = PROMPT_LENGTH

    regional_attention_mask = torch.zeros(
        (encoder_seq_len + hidden_seq_len, encoder_seq_len + hidden_seq_len),
        device=device,
        dtype=torch.bool,
    )

    # initialize self-attended mask
    self_attend_masks = torch.zeros((hidden_seq_len, hidden_seq_len), device=device, dtype=torch.bool)

    # initialize union mask
    union_masks = torch.zeros((hidden_seq_len, hidden_seq_len), device=device, dtype=torch.bool)

    # handle each mask
    for i in range(num_of_regions):
        # txt attends to itself
        regional_attention_mask[i*each_prompt_seq_len:(i+1)*each_prompt_seq_len, i*each_prompt_seq_len:(i+1)*each_prompt_seq_len] = True

        # txt attends to corresponding regional img
        regional_attention_mask[i*each_prompt_seq_len:(i+1)*each_prompt_seq_len, encoder_seq_len:] = masks[i].transpose(-1, -2)

        # regional img attends to corresponding txt
        regional_attention_mask[encoder_seq_len:, i*each_prompt_seq_len:(i+1)*each_prompt_seq_len] = masks[i]

        # regional img attends to corresponding regional img
        img_size_masks = masks[i][:, :1].repeat(1, hidden_seq_len)
        img_size_masks_transpose = img_size_masks.transpose(-1, -2)
        self_attend_masks = torch.logical_or(self_attend_masks,
                                             torch.logical_and(img_size_masks, img_size_masks_transpose))

        # update union
        union_masks = torch.logical_or(union_masks,
                                       torch.logical_or(img_size_masks, img_size_masks_transpose))

    background_masks = torch.logical_not(union_masks)

    background_and_self_attend_masks = torch.logical_or(background_masks, self_attend_masks)

    regional_attention_mask[encoder_seq_len:, encoder_seq_len:] = background_and_self_attend_masks
    return regional_attention_mask


def build_block_mask(dense_mask, device):
    """
    Turns a dense boolean attention mask into a flex-attention BlockMask.

    Regional Prompting builds its mask as a dense square; flex only needs to know, per 128 x 128 block, whether
    anything in it is allowed, which it works out by calling mask_mod on the index grid. Flex attention is what
    makes the regional sequence affordable: at 1024 px it is regions * 300 + 4096 tokens, so a dense 24-head score
    matrix is roughly a gigabyte per block.

    Args:
        dense_mask: [length, length] boolean tensor, True where attention is allowed.
        device: torch device the attention runs on.

    Returns:
        A BlockMask for a sequence of that length.
    """
    length = dense_mask.shape[0]
    dense_mask = dense_mask.to(device)

    def mask_mod(batch, head, query_index, key_index):
        return dense_mask[query_index, key_index]

    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=length, KV_LEN=length, device=device)


class RegionalPixelDiTPipeline(DiffusionPipeline):
    """Denoises one image under Regional Prompting control, with PixelDiT as the transformer."""

    def __init__(self, transformer, scheduler):
        """
        Args:
            transformer: RegionalPixDiT_T2I model.
            scheduler: DPMSolverMultistepScheduler configured for flow matching, with the checkpoint's flow shift.
        """
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler)

    @classmethod
    def from_weights(cls, device, weights_dir=None):
        """
        Loads the PixelDiT checkpoint into the regional model and builds PixelDiT's sampler.

        Args:
            device: torch device for the transformer.
            weights_dir: Path of a folder holding the checkpoint, or None to download it (Hugging Face cache).

        Returns:
            RegionalPixelDiTPipeline.
        """
        transformer, model_config = load_pixeldit(device, MODEL_DTYPE, weights_dir,
                                                  model_class=RegionalPixDiT_T2I)
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=NUM_TRAIN_TIMESTEPS,
            solver_order=SOLVER_ORDER,
            algorithm_type="dpmsolver++",
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            flow_shift=model_config["scheduler"]["flow_shift"],
        )
        return cls(transformer, scheduler)

    @torch.no_grad()
    def __call__(self, regional_embeds, regional_negative_embeds, base_embeds, base_negative_embeds,
                 regional_masks, cfg_scale, seed, mask_inject_steps, base_ratio, inject_blocks_interval):
        """
        Runs one generation.

        Args:
            regional_embeds: [regions, PROMPT_LENGTH, text dim] positive prompt of every region, in the order
                regional_masks is in.
            regional_negative_embeds: [regions, PROMPT_LENGTH, text dim] negative prompt of every region.
            base_embeds: [PROMPT_LENGTH, text dim] whole-image (background) positive prompt.
            base_negative_embeds: [PROMPT_LENGTH, text dim] whole-image negative prompt.
            regional_masks: list of [IMAGE_HEIGHT, IMAGE_WIDTH] tensors, 1 inside the region.
            cfg_scale: classifier-free guidance scale.
            seed: seed of the initial noise.
            mask_inject_steps: how many leading steps run under regional control.
            base_ratio: weight of the unmasked base stream in the image tokens.
            inject_blocks_interval: a patch block is masked iff its index % this == 0.

        Returns:
            [3, IMAGE_HEIGHT, IMAGE_WIDTH] float CPU tensor in [0, 1].
        """
        device = self.transformer.device

        regional_attention_mask = build_regional_attention_mask(regional_masks, device)
        block_mask = build_block_mask(regional_attention_mask, device)
        del regional_attention_mask

        regional_embeds = regional_embeds.to(device=device, dtype=MODEL_DTYPE).unsqueeze(0)
        regional_negative_embeds = regional_negative_embeds.to(device=device, dtype=MODEL_DTYPE).unsqueeze(0)
        base_embeds = base_embeds.to(device=device, dtype=MODEL_DTYPE)[None, None]
        base_negative_embeds = base_negative_embeds.to(device=device, dtype=MODEL_DTYPE)[None, None]

        generator = torch.Generator(device=device).manual_seed(seed)
        sample = randn_tensor((1, 3, IMAGE_HEIGHT, IMAGE_WIDTH), generator=generator, device=device)
        self.scheduler.set_timesteps(NUM_STEPS, device=device)

        for step_index, timestep in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # The model gets the exact time sigma * 1000 (scheduler.timesteps are rounded), in the model dtype.
            sigma = self.scheduler.sigmas[step_index].item()
            images = sample.to(MODEL_DTYPE)
            model_time = torch.full((1,), sigma * NUM_TRAIN_TIMESTEPS, device=device, dtype=MODEL_DTYPE)

            if step_index < mask_inject_steps:
                # Regional control: masked region prompts plus the unmasked base stream, on both CFG passes.
                positive = self.transformer(images, model_time, regional_embeds,
                                            text_embeds_base=base_embeds, base_ratio=base_ratio,
                                            regional_attention_mask=block_mask,
                                            inject_blocks_interval=inject_blocks_interval).float()
                negative = self.transformer(images, model_time, regional_negative_embeds,
                                            text_embeds_base=base_negative_embeds, base_ratio=base_ratio,
                                            regional_attention_mask=block_mask,
                                            inject_blocks_interval=inject_blocks_interval).float()
            else:
                # Plain generation on the background prompt, exactly as the FLUX original reverts to.
                positive = self.transformer(images, model_time, base_embeds).float()
                negative = self.transformer(images, model_time, base_negative_embeds).float()

            velocity = negative + cfg_scale * (positive - negative)
            sample = self.scheduler.step(velocity, timestep, sample).prev_sample

        return to_display_image(sample[0])
