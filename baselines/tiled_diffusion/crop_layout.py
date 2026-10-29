"""
Turning a config into the tiles tiled-diffusion generates, and into the canvas they compose.

A crop is a region of the final image and a tile is one candidate that could fill it. This baseline knows only
about tiles and how their sides connect, so "candidate" is expressed through the connections alone: every tile of
a crop is given the SAME side_id / side_dir pair and differs only in its prompt. generate_graph_groups matches
sides by (side_id, opposite side_dir) and apply_similarity_constraint then forces every matched side to carry the
same band (latent_handler.py), which is exactly what makes the candidates of a crop interchangeable at the seam.

Every tile is generated as one full-size square, and the crops are laid out in a near-square grid, the same grid
mnm_baseline's whole images are assembled into (metrics/composites.py: gallery, and the user study's
build_study.gallery): ceil(sqrt(n)) columns, as many rows as that needs, filled row by row with the crops in
reading order - their config rectangles top to bottom, then left to right. The config's crop rectangles decide
that order only, not the geometry: a combination is columns * IMAGE_WIDTH wide by rows * IMAGE_HEIGHT tall, and
the cells the crops do not fill (3, 5, 7 or 8 crops) stay empty. That is the baseline as it was run - each model
generates at the size it was released for, and for PixelDiT that is one 1024 x 1024 square per crop.

Sides are indexed [right, left, up, down]. Every pair of crops in neighbouring cells shares one junction: a
horizontal one joins the left crop's RIGHT side ('cw') to the right crop's LEFT side ('ccw'), a vertical one the
upper crop's DOWN side ('cw') to the lower crop's UP side ('ccw'). For 4 crops:

                 column 0                          column 1
    row 0   crop a  R=1 cw, D=2 cw    -- 1 --   crop b  L=1 ccw, D=3 cw
                   |                                   |
                   2                                   3
                   |                                   |
    row 1   crop c  U=2 ccw, R=4 cw   -- 4 --   crop d  U=3 ccw, L=4 ccw

A side with no crop beyond it (the grid's border, or an empty cell) stays None and unconstrained. With a single
crop nothing connects to anything and each tile is an ordinary unconstrained generation.

Every tile is padded on the same axes, whichever of its own sides connect: the tiling constraint copies a band of
one tile straight into another's padding, which needs the two to have the same shape. So a single row pads left
and right, a single column top and bottom, and a grid of both all four sides; the padding is cut away again when
the image is read out (model.py).

Copied from tiled-diffusion's crop_layout.py, keeping the forced-square path only.
"""

import math

from macros import IMAGE_HEIGHT, IMAGE_WIDTH

from .latent_class import LatentClass

# Side indices, as used by LatentClass.side_id / side_dir.
RIGHT, LEFT, UP, DOWN = 0, 1, 2, 3


class PlacedCrop:
    """A crop, its cell in the grid, and the box its tile occupies in a combination."""

    def __init__(self, index, position, columns):
        """
        Args:
            index: the crop's index in the config's lists.
            position: its place in reading order, which fills the grid row by row.
            columns: the grid's column count.
        """
        self.index = index
        self.position = position
        self.row, self.column = divmod(position, columns)
        self.place_x = self.column * IMAGE_WIDTH
        self.place_y = self.row * IMAGE_HEIGHT
        self.tile_indices = []  # positions of this crop's candidates in latents_arr


def grid_shape(num_crops):
    """
    Args:
        num_crops: how many crops the grid holds.

    Returns:
        (rows, columns): ceil(sqrt(n)) columns and as many rows as the crops need, as in mnm_baseline's gallery.
    """
    columns = math.ceil(math.sqrt(num_crops))
    return math.ceil(num_crops / columns), columns


def canvas_size(num_crops):
    """
    Args:
        num_crops: how many crops the grid holds.

    Returns:
        (width, height) of a combination in pixels.
    """
    rows, columns = grid_shape(num_crops)
    return columns * IMAGE_WIDTH, rows * IMAGE_HEIGHT


def reading_order(config):
    """
    Args:
        config: MixNMatchConfig.

    Returns:
        Crop indices ordered by their config rectangles: top to bottom, then left to right, as mnm_baseline's
        gallery orders them. A config without rectangles (dynamic cropping) keeps its listed order.
    """
    if not config.crops:
        return list(range(config.num_crops))
    return sorted(range(config.num_crops), key=lambda index: (config.crops[index]["y"], config.crops[index]["x"]))


def build_latents(config):
    """
    Builds every candidate tile of every crop, wired to its neighbours in the grid.

    Args:
        config: MixNMatchConfig.

    Returns:
        (latents_arr, placed_crops): the flat list of LatentClass tiles handed to the model, and the crops in
        reading order, each recording which positions in that list are its own candidates. latents_arr is in
        crop-major order of the config's crop indices, which is the order model.set_text_embeddings expects.
    """
    rows, columns = grid_shape(config.num_crops)
    placed_crops = [PlacedCrop(crop_index, position, columns)
                    for position, crop_index in enumerate(reading_order(config))]
    by_cell = {(placed.row, placed.column): placed for placed in placed_crops}

    # One junction id per pair of neighbouring cells, counted from 1, horizontal and vertical alike.
    junctions = {}
    for placed in placed_crops:
        for neighbour_cell in ((placed.row, placed.column + 1), (placed.row + 1, placed.column)):
            if neighbour_cell in by_cell:
                junctions[(placed.row, placed.column), neighbour_cell] = len(junctions) + 1

    # Every tile carries the same padding (see the module docstring), so the tiling copies line up.
    pad_x, pad_y = columns > 1, rows > 1

    latents_arr = [None] * (config.num_crops * config.tiles_per_crop)
    for placed in placed_crops:
        cell = (placed.row, placed.column)
        side_id, side_dir = [None, None, None, None], [None, None, None, None]
        # This crop is the first cell of the junctions to its right and below it, the second of those to its
        # left and above it.
        for side, neighbour_cell, direction in ((RIGHT, (placed.row, placed.column + 1), 'cw'),
                                                (DOWN, (placed.row + 1, placed.column), 'cw'),
                                                (LEFT, (placed.row, placed.column - 1), 'ccw'),
                                                (UP, (placed.row - 1, placed.column), 'ccw')):
            key = (cell, neighbour_cell) if direction == 'cw' else (neighbour_cell, cell)
            if key in junctions:
                side_id[side], side_dir[side] = junctions[key], direction

        for tile_index in range(config.tiles_per_crop):
            flat_index = placed.index * config.tiles_per_crop + tile_index
            placed.tile_indices.append(flat_index)
            latents_arr[flat_index] = LatentClass(
                prompt=config.tile_prompts[placed.index][tile_index],
                negative_prompt=config.tile_negative_prompts[placed.index][tile_index],
                height=IMAGE_HEIGHT,
                width=IMAGE_WIDTH,
                # Every candidate of this crop gets its own copy of the same descriptors.
                side_id=list(side_id),
                side_dir=list(side_dir),
                pad_x=pad_x,
                pad_y=pad_y,
            )
    return latents_arr, placed_crops


def describe(config, latents_arr, placed_crops):
    """
    Renders the layout and the wiring as a table, for --dry-run.

    Args:
        config: MixNMatchConfig.
        latents_arr: the tiles from build_latents.
        placed_crops: the crops from build_latents.

    Returns:
        The table as one string.
    """
    rows, columns = grid_shape(config.num_crops)
    width, height = canvas_size(config.num_crops)
    lines = [f"{config.num_crops} crops x {config.tiles_per_crop} candidates = {len(latents_arr)} tiles, "
             f"in a {rows} x {columns} grid composed into {width}x{height}", ""]
    lines.append(f"{'idx':>4}  {'crop':>4}  {'cell':>6}  {'cand':>4}  {'side_id (R,L,U,D)':<22}  "
                 f"{'side_dir (R,L,U,D)':<30}  prompt")
    lines.append("-" * 120)
    for placed in placed_crops:
        for candidate_index, flat_index in enumerate(placed.tile_indices):
            latent = latents_arr[flat_index]
            cell = f"{placed.row},{placed.column}"
            lines.append(f"{flat_index:>4}  {placed.index:>4}  {cell:>6}  {candidate_index:>4}  "
                         f"{str(latent.side_id):<22}  {str(latent.side_dir):<30}  {latent.prompt}")
        lines.append("")
    return "\n".join(lines)
