"""
The tiled-diffusion denoising run, with PixelDiT as the model.

Every tile is denoised on its own, one after the other, and between two steps the constraints of
latent_handler.py rewrite parts of all of them: the similarity constraint makes the candidates of a crop agree
just inside their shared side, and the tiling constraint fills each tile's padding with the neighbouring crop's
image. After the last step one closing padding pass runs, and the padding is cut away.

Copied from tiled-diffusion's model.py (class PixelDiTLatentTiling), with its Gemma encoding replaced by the
root's encode_all_prompts and its model loading by the root's load_pixeldit. The differences from the Stable
Diffusion original it was adapted from are all consequences of PixelDiT being a pixel space model:

  * a tile's tensor is the 3 channel image itself, at full resolution, so there is no VAE to encode or decode
    through;
  * conditioning comes from Gemma-2-2b-it, encoded before the transformer is loaded, because the two do not fit
    on an 8 GB card together;
  * sampling is DPM-Solver++ (order 2) on flow sigmas. That solver is multistep: it carries a history and a step
    index, so unlike DDPM it cannot be shared between tiles and every tile gets its own;
  * max_width / max_replica_width arrive counted in 16 pixel patches and are turned into pixels here, so the
    LatentHandler band copies keep working unchanged.
"""

import gc
import logging

import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler
from tqdm.auto import tqdm

from macros import NUM_STEPS, PATCH_SIZE_PIXELS
from pipeline_mix_n_match import MODEL_DTYPE, NUM_TRAIN_TIMESTEPS, SOLVER_ORDER
from pixeldit_model import load_pixeldit

from .latent_handler import LatentHandler, generate_graph_groups


def set_text_embeddings(latents_arr, encoded_prompts):
    """
    Gives every tile the prompt features it is denoised with.

    Args:
        latents_arr: the LatentClass tiles, in crop-major order, which is the order of the config's tile prompts.
        encoded_prompts: the config's EncodedPrompts; prompt 0 is the background prompt, which this baseline
            does not use, and prompt 1 + crop * k + tile is that tile's.
    """
    for index, latent in enumerate(latents_arr):
        latent.text_embeddings = encoded_prompts.embeddings[:, index + 1]


class PixelDiTLatentTiling:
    """The tiled-diffusion run: every tile of every crop, denoised together under the constraints."""

    def __init__(self, weights_dir=None):
        """
        Args:
            weights_dir: Path of a folder holding the checkpoint, or None to download it (Hugging Face cache).
        """
        # The transformer is loaded in __call__, after the prompts are encoded and Gemma is freed.
        self.weights_dir = weights_dir
        self.transformer = None
        self.model_config = None

    def build_scheduler(self, device):
        """
        Builds one tile's sampler: PixelDiT's inference settings, with the checkpoint's flow shift.

        Args:
            device: torch device the timesteps are placed on.

        Returns:
            A DPMSolverMultistepScheduler with its timesteps already set.
        """
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=NUM_TRAIN_TIMESTEPS,
            solver_order=SOLVER_ORDER,
            algorithm_type="dpmsolver++",
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            flow_shift=self.model_config["scheduler"]["flow_shift"],
        )
        scheduler.set_timesteps(NUM_STEPS, device=device)
        return scheduler

    def predict_velocity(self, latent, sigma, cfg_scale, device):
        """
        Runs the transformer for both guidance passes on one tile and combines them.

        Args:
            latent: the LatentClass tile, denoised from its post_latent.
            sigma: current noise level.
            cfg_scale: classifier-free guidance scale.
            device: torch device the model runs on.

        Returns:
            [1, 3, H, W] float32 flow velocity, the shape of the tile's own tensor.
        """
        # The model takes a stack of images, [batch, S, 3, H, W]; one tile at a time means batch = S = 1.
        images = latent.post_latent.unsqueeze(0).to(MODEL_DTYPE)
        # The exact time sigma * 1000 (scheduler.timesteps are rounded), in the model dtype.
        model_time = torch.full((1,), sigma * NUM_TRAIN_TIMESTEPS, device=device, dtype=MODEL_DTYPE)
        # [2, prompt length, text dim] -> [2, 1, prompt length, text dim]: one prompt per guidance pass.
        text_embeds = latent.text_embeddings.unsqueeze(1).to(device=device, dtype=MODEL_DTYPE)
        # Sequential classifier free guidance: the two passes one after the other give the same result as one
        # batch of two for about half the memory, which is what fits next to the 1.3B checkpoint on 8 GB.
        negative = self.transformer(images, model_time, text_embeds[:1], None)[0].float()
        positive = self.transformer(images, model_time, text_embeds[1:], None)[0].float()
        return negative + cfg_scale * (positive - negative)

    @torch.no_grad()
    def __call__(self, latents_arr, encoded_prompts, config, max_width, max_replica_width, device):
        """
        Denoises every tile of every crop under the constraints and stores each one's finished image.

        Args:
            latents_arr: the LatentClass tiles from crop_layout.build_latents.
            encoded_prompts: the config's EncodedPrompts.
            config: MixNMatchConfig (seed and cfg_scale).
            max_width: seam context, in 16 pixel patches.
            max_replica_width: width of the similarity band, in the same units.
            device: torch device the model runs on.

        Returns:
            latents_arr, every tile carrying its image as a [height, width, 3] array in [0, 1].
        """
        # The widths arrive in 16 pixel patches, but the LatentHandler methods slice the tensor itself, which
        # here is the image, so from this point on they are pixels.
        pad = max_width * PATCH_SIZE_PIXELS
        replica = max_replica_width * PATCH_SIZE_PIXELS

        set_text_embeddings(latents_arr, encoded_prompts)
        self.transformer, self.model_config = load_pixeldit(device, MODEL_DTYPE, self.weights_dir)
        logging.warning("Finished loading models..")

        generator = torch.Generator(device='cuda')
        generator.manual_seed(config.seed)
        schedulers = []
        for latent in latents_arr:
            latent.set_latents(generator=generator, max_width=pad)
            schedulers.append(self.build_scheduler(device))
        # Every tile's sampler was built the same way, so any of them can drive the loop.
        timesteps, sigmas = schedulers[0].timesteps, schedulers[0].sigmas

        graph_groups = generate_graph_groups(latents_arr)
        for i, t in tqdm(enumerate(timesteps)):
            logging.warning(f"Running step {i} of {NUM_STEPS}")
            logging.warning("Applying Similarity Constraint")
            latents_arr = LatentHandler.apply_similarity_constraint(latents_arr, i,
                                                                    groups=graph_groups,
                                                                    max_width=pad,
                                                                    max_replica_width=replica)
            for latent in latents_arr:
                latent.clone_post_latents()
            logging.warning("Tiling latents")
            latents_arr = LatentHandler.tile(latents_arr, i, groups=graph_groups, max_width=pad)
            sigma = sigmas[i].item()
            for latent, scheduler in zip(latents_arr, schedulers):
                torch.cuda.empty_cache()
                gc.collect()
                # Flow matching: the model predicts the velocity, noise - image, rather than the noise.
                velocity = self.predict_velocity(latent, sigma, config.cfg_scale, device)
                latent.pre_latent = scheduler.step(velocity, t, latent.post_latent).prev_sample
                latent.clone_post_latents()

        logging.warning("Getting images")
        logging.warning("Applying Similarity Constraint")
        latents_arr = LatentHandler.apply_similarity_constraint(latents_arr, NUM_STEPS,
                                                                groups=graph_groups,
                                                                max_width=pad,
                                                                max_replica_width=replica)
        for latent in latents_arr:
            latent.clone_post_latents()
        logging.warning("Tiling latents")
        latents_arr = LatentHandler.tile(latents_arr, NUM_STEPS, groups=graph_groups, max_width=pad)

        logging.warning("Applying Random Padding Constraint")
        latents_arr = LatentHandler.apply_random_padding_constraint(latents_arr, groups=graph_groups,
                                                                    max_width=pad)

        for latent in latents_arr:
            torch.cuda.empty_cache()
            gc.collect()
            # Pixel space: the tensor is the image already, so there is no VAE decode, only the [-1, 1] to
            # [0, 1] rescale and the crop that takes the padding back off.
            image_t = ((latent.post_latent[0].float().clamp(-1, 1) + 1) / 2).cpu().numpy()
            img_t_rgb = np.transpose(image_t, (1, 2, 0))
            if latent.is_xy():
                img_t_rgb = img_t_rgb[pad:-pad, pad:-pad, :]
            elif latent.is_x():
                img_t_rgb = img_t_rgb[:, pad:-pad, :]
            elif latent.is_y():
                img_t_rgb = img_t_rgb[pad:-pad, :, :]

            latent.image = img_t_rgb

        return latents_arr
