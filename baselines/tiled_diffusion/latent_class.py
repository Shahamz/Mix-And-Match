"""
One candidate tile of the tiled-diffusion baseline: its prompts, its noisy image, and how its sides connect.

A tile carries two copies of its image: pre_latent, what the sampler produced, and post_latent, that copy with the
constraint bands written into it (latent_handler.py). The model denoises post_latent and writes the result back to
pre_latent, so a step always reads what the constraints last agreed on.

The image is padded on the axes along which the crops' grid has junctions - left and right when it has more than
one column, top and bottom when it has more than one row - the same for every tile, so a band copied from one tile
fits another's padding (crop_layout.py). The padding is cut away again when the finished image is read out
(model.py).

Copied from tiled-diffusion's latent_class.py, keeping only what the PixelDiT path uses: the tensor is the image
itself, so there is no latent downsampling and no CLIP text encoding on the tile.
"""

import torch

# Channels of a PixelDiT sample: it denoises the image itself, not a latent.
PIXEL_CHANNELS = 3


class LatentClass:
    """One candidate tile of one crop."""

    def __init__(self, prompt, negative_prompt, height, width, side_id, side_dir, pad_x, pad_y):
        """
        Args:
            prompt: the tile's positive prompt.
            negative_prompt: its negative prompt.
            height, width: size the tile is generated at, in pixels.
            side_id: junction id of each side [right, left, up, down]; None where the side connects to nothing.
            side_dir: 'cw' or 'ccw' per side, the orientation that makes two facing sides match.
            pad_x: whether the tile is padded left and right.
            pad_y: whether the tile is padded top and bottom.
        """
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.height = height
        self.width = width
        self.side_id = side_id
        self.side_dir = side_dir
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.text_embeddings = None  # [2, prompt length, text dim]: index 0 negative, index 1 positive
        self.pre_latent = None
        self.post_latent = None
        self.image = None  # [height, width, 3] float array in [0, 1], set when the run finishes

    def is_xy(self):
        """
        Returns:
            True when the tile is padded on all four sides.
        """
        return self.pad_x and self.pad_y

    def is_x(self):
        """
        Returns:
            True when the tile is padded left and right.
        """
        return self.pad_x

    def is_y(self):
        """
        Returns:
            True when the tile is padded top and bottom.
        """
        return self.pad_y

    def set_latents(self, generator, max_width):
        """
        Draws the tile's initial noise, with room for the padding its connected sides need.

        Args:
            generator: torch generator, shared by every tile so the whole run follows one seed.
            max_width: padding added on each connected side, in pixels.
        """
        height, width = self.height, self.width
        if self.is_xy():
            self.pre_latent = torch.randn(
                (1, PIXEL_CHANNELS, height + 2 * max_width, width + 2 * max_width),
                generator=generator,
                device='cuda',
            )
        elif self.is_x():
            self.pre_latent = torch.randn(
                (1, PIXEL_CHANNELS, height, width + 2 * max_width),
                generator=generator,
                device='cuda',
            )
        elif self.is_y():
            self.pre_latent = torch.randn(
                (1, PIXEL_CHANNELS, height + 2 * max_width, width),
                generator=generator,
                device='cuda',
            )
        else:
            self.pre_latent = torch.randn(
                (1, PIXEL_CHANNELS, height, width),
                generator=generator,
                device='cuda',
            )
        self.clone_post_latents()

    def clone_post_latents(self):
        """Copies pre_latent into post_latent, which is what the constraints write into and the model denoises."""
        self.post_latent = self.pre_latent.clone()
