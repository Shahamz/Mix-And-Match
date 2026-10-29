"""
The three constraints tiled-diffusion applies between denoising steps, and the wiring they run over.

The crops of the run form a grid: side `s` of one tile carries the same junction id as the side facing it on the
neighbouring crop's tiles, with the opposite orientation. generate_graph_groups turns that into groups - for every
(tile, side) the list of (tile, side) pairs facing it, which are the candidates of the neighbouring crop.

    tile                         copies a band of the neighbour's image into this tile's padding, so a tile is
                                 denoised as if the neighbour continued past its edge. The neighbour it copies
                                 from rotates with the step, so over a run every candidate is seen.
    apply_similarity_constraint  forces the candidates of one crop to carry the SAME band just inside their
                                 shared side, which is what makes them interchangeable there.
    apply_random_padding_constraint  one closing pass over the padding, from a randomly chosen candidate. It only
                                 touches padding, which is cut away before the image is saved.

Copied from tiled-diffusion's latent_handler.py, with generate_graph_groups and
get_tensor_slice_from_latent_and_side_idx from its utils.py and the rotation matrices from its config.py, since
those files hold nothing else this baseline uses. Sides are indexed [right, left, up, down].
"""

import random

import torch

# Rotation (in quarter turns) that carries a band of side i onto side j, for the similarity and tiling copies.
SIMILARITY_ROTATION_MATRIX = [
    [0, 2, 1, 3],
    [2, 0, 3, 1],
    [3, 1, 0, 2],
    [1, 3, 2, 0]
]

TILING_ROTATION_MATRIX = [
    [2, 0, 1, 3],
    [0, 2, 3, 1],
    [3, 1, 2, 0],
    [1, 3, 0, 2]
]


def generate_graph_groups(latents_arr):
    """
    Finds, for every (tile, side) that carries a junction, the (tile, side) pairs facing it.

    Two sides face each other when they carry the same side_id with opposite side_dir, which is how
    crop_layout.py wires the grid: every candidate of one crop faces every candidate of each neighbouring crop.

    Args:
        latents_arr: the LatentClass tiles.

    Returns:
        {"<tile index>_<side index>": {"target_latent_idx": [...], "target_side_idx": [...]}}.
    """
    groups = {}
    for latent_idx, latent in enumerate(latents_arr):
        side_id = latent.side_id
        side_dir = latent.side_dir
        for side_idx, side_val in enumerate(side_id):
            if side_val is None:
                continue
            current_side_dir = side_dir[side_idx]
            if current_side_dir == 'cw':
                opposite_side_dir = 'ccw'
            else:
                opposite_side_dir = 'cw'
            for latent2_idx, latent2 in enumerate(latents_arr):
                side_id_2 = latent2.side_id
                side_dir_2 = latent2.side_dir
                for target_side_idx, targets in enumerate(zip(side_id_2, side_dir_2)):
                    target_side, target_dir = targets
                    if target_side == side_val and target_dir == opposite_side_dir:
                        key = f"{latent_idx}_{side_idx}"
                        if key not in groups:
                            groups[key] = {'target_latent_idx': [], 'target_side_idx': []}
                        groups[key]['target_latent_idx'].append(latent2_idx)
                        groups[key]['target_side_idx'].append(target_side_idx)
    return groups


def get_tensor_slice_from_latent_and_side_idx(latent, side_idx, max_width, max_replica_width):
    """
    Reads the band of width max_replica_width that sits just inside one side's padding.

    Args:
        latent: [1, 3, H, W] tile tensor.
        side_idx: which side, [right, left, up, down].
        max_width: the padding width on that side, in pixels.
        max_replica_width: how wide a band to read.

    Returns:
        [1, 3, ...] view of that band.
    """
    B, F, H, W = latent.size()
    if side_idx == 0:  # Right
        tensor = latent[:, :, :, W - max_width - max_replica_width: W - max_width]
    elif side_idx == 1:  # Left
        tensor = latent[:, :, :, max_width:max_width + max_replica_width]
    elif side_idx == 2:  # Up
        tensor = latent[:, :, max_width:max_width + max_replica_width, :]
    else:  # Down
        tensor = latent[:, :, H - max_width - max_replica_width: H - max_width, :]
    return tensor


class LatentHandler:
    """The constraints, each rewriting part of every tile between two denoising steps."""

    @staticmethod
    def tile(latents_arr, step, groups, max_width=10):
        """
        Fills every tile's padding with the neighbouring crop's image, so each tile is denoised as if the
        neighbour continued past its edge.

        Which candidate of the neighbouring crop is copied rotates with the step, so a tile is not fitted to one
        particular neighbour.

        Args:
            latents_arr: the LatentClass tiles.
            step: the denoising step, which picks the neighbour.
            groups: the wiring from generate_graph_groups.
            max_width: padding width in pixels.

        Returns:
            latents_arr, with post_latent updated.
        """
        for key, val in groups.items():
            target_latent_idx = val['target_latent_idx']
            target_side_idx = val['target_side_idx']
            len_target = len(target_latent_idx)
            latent_source_id, side_source_id = map(int, key.split('_'))
            candidate_idx = step % len_target
            chosen_latent_idx = target_latent_idx[candidate_idx]
            chosen_side_idx = target_side_idx[candidate_idx]
            source_latent = latents_arr[latent_source_id]
            target_latent = latents_arr[chosen_latent_idx]
            B, F, H, W = source_latent.pre_latent.size()
            if chosen_side_idx == 0:  # Right
                tensor = target_latent.pre_latent[:, :, :, W - 2 * max_width: W - max_width]
            elif chosen_side_idx == 1:  # Left
                tensor = target_latent.pre_latent[:, :, :, max_width: 2 * max_width]
            elif chosen_side_idx == 2:  # Up
                tensor = target_latent.pre_latent[:, :, max_width: 2 * max_width, :]
            else:  # Down
                tensor = target_latent.pre_latent[:, :, H - 2 * max_width: H - max_width, :]
            rotation_value = TILING_ROTATION_MATRIX[side_source_id][chosen_side_idx]
            rotated_tensor = torch.rot90(tensor, rotation_value, [2, 3])

            if side_source_id == 0:  # Right
                source_latent.post_latent[:, :, :, W - max_width:] = rotated_tensor
            elif side_source_id == 1:  # Left
                source_latent.post_latent[:, :, :, :max_width] = rotated_tensor
            elif side_source_id == 2:  # Up
                source_latent.post_latent[:, :, :max_width, :] = rotated_tensor
            else:  # Down
                source_latent.post_latent[:, :, H - max_width:, :] = rotated_tensor

        return latents_arr

    @staticmethod
    def apply_similarity_constraint(latents_arr, step, groups, max_width=10, max_replica_width=5):
        """
        Gives every candidate of one crop the same band just inside a shared side, so they agree at the junction.

        The band is taken from one of them, which rotates with the step.

        Args:
            latents_arr: the LatentClass tiles.
            step: the denoising step, which picks the candidate the band comes from.
            groups: the wiring from generate_graph_groups.
            max_width: padding width in pixels.
            max_replica_width: width of the band, in pixels.

        Returns:
            latents_arr, with pre_latent updated.
        """
        for key, val in groups.items():
            target_latent_idx = val['target_latent_idx']
            target_side_idx = val['target_side_idx']
            len_target = len(target_latent_idx)
            if len_target <= 1:
                continue
            candidate_idx = step % len_target
            source_latent_idx = target_latent_idx[candidate_idx]
            source_side_idx = target_side_idx[candidate_idx]
            latent = latents_arr[source_latent_idx]
            tensor_slice = get_tensor_slice_from_latent_and_side_idx(latent=latent.pre_latent,
                                                                     side_idx=source_side_idx,
                                                                     max_width=max_width,
                                                                     max_replica_width=max_replica_width)
            for lat_idx, side_idx in zip(target_latent_idx, target_side_idx):
                if source_latent_idx == lat_idx and source_side_idx == side_idx:
                    continue
                rotation_value = SIMILARITY_ROTATION_MATRIX[source_side_idx][side_idx]
                rotated_tensor_slice = torch.rot90(tensor_slice, rotation_value, [2, 3])
                current_latent = latents_arr[lat_idx]
                B, F, H, W = latent.pre_latent.size()
                if side_idx == 0:  # Right
                    current_latent.pre_latent[:, :, :,
                    W - max_width - max_replica_width: W - max_width] = rotated_tensor_slice
                elif side_idx == 1:  # Left
                    current_latent.pre_latent[:, :, :, max_width:max_width + max_replica_width] = rotated_tensor_slice
                elif side_idx == 2:  # Up
                    current_latent.pre_latent[:, :, max_width:max_width + max_replica_width, :] = rotated_tensor_slice
                else:  # Down
                    current_latent.pre_latent[:, :, H - max_width - max_replica_width: H - max_width,
                    :] = rotated_tensor_slice
        return latents_arr

    @staticmethod
    def apply_random_padding_constraint(latents_arr, groups, max_width=10):
        """
        The closing pass: every tile's padding on a junction is overwritten from one randomly chosen candidate of
        the neighbouring crop.

        It runs after the last denoising step and touches the padding only, which is cut away before the image is
        saved, so the choice does not change what is written.

        Args:
            latents_arr: the LatentClass tiles.
            groups: the wiring from generate_graph_groups.
            max_width: padding width in pixels.

        Returns:
            latents_arr, with post_latent updated.
        """
        for key, val in groups.items():
            target_latent_idx = val['target_latent_idx']
            target_side_idx = val['target_side_idx']
            len_target = len(target_latent_idx)

            if len_target <= 1:
                continue

            # Randomly choose one of the targets
            random_idx = random.randint(0, len_target - 1)
            source_latent_idx = target_latent_idx[random_idx]
            source_side_idx = target_side_idx[random_idx]

            latent = latents_arr[source_latent_idx]
            tensor_slice = get_tensor_slice_from_latent_and_side_idx(latent=latent.post_latent,
                                                                     side_idx=source_side_idx,
                                                                     max_width=0,
                                                                     max_replica_width=max_width)

            for lat_idx, side_idx in zip(target_latent_idx, target_side_idx):
                if source_latent_idx == lat_idx and source_side_idx == side_idx:
                    continue

                rotation_value = SIMILARITY_ROTATION_MATRIX[source_side_idx][side_idx]
                rotated_tensor_slice = torch.rot90(tensor_slice, rotation_value, [2, 3])
                current_latent = latents_arr[lat_idx]
                B, F, H, W = latent.post_latent.size()

                if side_idx == 0:  # Right
                    current_latent.post_latent[:, :, :, W - max_width:] = rotated_tensor_slice
                elif side_idx == 1:  # Left
                    current_latent.post_latent[:, :, :, :max_width] = rotated_tensor_slice
                elif side_idx == 2:  # Up
                    current_latent.post_latent[:, :, :max_width, :] = rotated_tensor_slice
                else:  # Down
                    current_latent.post_latent[:, :, H - max_width:, :] = rotated_tensor_slice

        return latents_arr
