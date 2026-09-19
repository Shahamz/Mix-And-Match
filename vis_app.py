"""
Saving and viewing Mix-And-Match results.

A crop is any set of patches (a rectangle, or a region of the saliency segmentation); a crop map gives the crop of
every patch. A combination is one tile per crop, pasted through the crop's mask; since the crops cover the image
exactly, every combination is a complete, gap-free image. With k tiles per crop and n non-empty crops there are k^n
combinations (the background crop of dynamic cropping has a single tile, shared by all of them).

Saving (called by generate.py): everything lives in outputs/, or in outputs/<config folder name>/ when a whole
folder of configs is run, so one sweep keeps its results together.
    - save_outputs: cuts the pipeline output into tiles and writes what the config asks for into the run folder
      <prefix>_<num_crops>crops_<tiles_per_crop>tiles/:
            visSheet.png (one sheet) or visSheets/part<i>of<N>.png (several), with "visualize";
            saliency/c_bg.png and saliency/c<i>_t<j>.png (per prompt), saliency/crop<i>.png and
            saliency/crop_bg.png (per crop) and saliency/crop_segmentation.png, with "save_saliency_maps" and
            dynamic cropping (saliency.py);
            tiles/crop<i>/tile<j>.png (the crop's bounding box, transparent outside the crop), tiles/cropBg with
            the background crop's single tile when there is one, plus tiles/layout.json (image size, tiles per
            crop, crop map, background crop), read by the viewer, with "save_separate_tiles".
    - save_standard_run: the single image of a standard run (generate.py --std) as <prefix>_std_run.png.

Viewer app (a tiles folder is its input):
    python vis_app.py outputs/<prefix>_<n>crops_<k>tiles/tiles
    Left click on a crop shows its next tile, right click the previous one, and Ctrl+S saves the shown combination
    to <tiles folder>/saved_combinations/.
"""

import argparse
import itertools
import json
import math
import sys
import tkinter as tk
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
from torchvision.transforms.functional import to_pil_image

from macros import IMAGE_HEIGHT, IMAGE_WIDTH, PATCH_SIZE_PIXELS, save_image
from saliency import save_saliency_outputs

# Maximum number of tile combinations drawn on one visualization sheet; more combinations start a new sheet.
MAX_COMBINATIONS_PER_SHEET = 16

# File in a tiles folder that describes the layout (image size, tiles per crop, crop map).
LAYOUT_FILENAME = "layout.json"

# Subfolder of a tiles folder where the viewer saves combinations (Ctrl+S).
SAVED_COMBINATIONS_DIRNAME = "saved_combinations"

# Names inside a run folder: the tiles folder, a lone sheet, the folder of several sheets, and the saliency folder.
TILES_DIRNAME = "tiles"
SINGLE_SHEET_FILENAME = "visSheet.png"
SHEETS_DIRNAME = "visSheets"
SALIENCY_DIRNAME = "saliency"

# Sheet drawing: white gap around combinations, height of the label strip under each, and its font size.
SHEET_GAP_PIXELS = 16
SHEET_LABEL_HEIGHT = 32
SHEET_LABEL_FONT_SIZE = 22

# Largest side, in screen pixels, of the image shown by the viewer; bigger images are scaled down to fit.
MAX_DISPLAY_SIZE = 900

def crop_regions(crop_map, num_crops):
    """
    Finds where every crop lies in pixels.

    Args:
        crop_map: [patch rows, patch cols] array of crop indices.
        num_crops: n.

    Returns:
        List with, per crop, (box, alpha) or None when the crop covers no patch. box is the (left, top, right,
        bottom) pixel bounding box; alpha is a PIL "L" image of the box, 255 inside the crop and 0 outside.
    """
    pixel_map = np.asarray(crop_map).repeat(PATCH_SIZE_PIXELS, axis=0).repeat(PATCH_SIZE_PIXELS, axis=1)
    regions = []
    for crop_index in range(num_crops):
        rows, cols = np.nonzero(pixel_map == crop_index)
        if len(rows) == 0:
            regions.append(None)
            continue
        left, top, right, bottom = int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1
        inside = pixel_map[top:bottom, left:right] == crop_index
        regions.append(((left, top, right, bottom), Image.fromarray(inside.astype(np.uint8) * 255)))
    return regions


def cut_tile(image, region):
    """
    Args:
        image: PIL RGB full image.
        region: (box, alpha) from crop_regions.

    Returns:
        PIL RGBA image of the box, transparent outside the crop.
    """
    box, alpha = region
    tile = image.crop(box).convert("RGBA")
    tile.putalpha(alpha)
    return tile


def tile_combinations(crop_tiles):
    """
    Args:
        crop_tiles: crop_tiles[i] is the list of tile images of crop i (empty for a crop that covers no patch).

    Returns:
        List of every combination: a tile index per crop (always 0 for an empty crop).
    """
    return list(itertools.product(*(range(max(len(tiles), 1)) for tiles in crop_tiles)))


def assemble_combination(crop_tiles, regions, combination, image_size):
    """
    Pastes one tile per crop into a full image, through the crops' masks.

    Args:
        crop_tiles: crop_tiles[i][j] is the RGBA image of tile j of crop i.
        regions: crop_regions output.
        combination: tile index per crop.
        image_size: (width, height) of the full image.

    Returns:
        PIL RGB image.
    """
    image = Image.new("RGB", image_size)
    for tiles, region, tile_index in zip(crop_tiles, regions, combination, strict=True):
        if region is not None:
            tile = tiles[tile_index]
            image.paste(tile, region[0][:2], tile)  # the tile's alpha keeps the pixels outside its crop
    return image


def crop_folder_name(crop_index, background_crop):
    """
    Args:
        crop_index: index of the crop.
        background_crop: index of the background crop, or None.

    Returns:
        "cropBg" for the background crop, "crop<i>" otherwise.
    """
    return "cropBg" if crop_index == background_crop else f"crop{crop_index}"


def combination_label(combination, crop_tiles, background_crop):
    """
    Args:
        combination: tile index per crop.
        crop_tiles: crop_tiles[i] is the list of tiles of crop i (empty crops are left out of the label).
        background_crop: index of the background crop, or None; it is the same in every combination, so it is
            left out too.

    Returns:
        Text such as "c0:t1  c1:t0" (crop i shows tile j).
    """
    return "  ".join(f"c{crop_index}:t{tile_index}"
                     for crop_index, (tile_index, tiles) in enumerate(zip(combination, crop_tiles))
                     if tiles and crop_index != background_crop)


def save_visualization_sheets(crop_tiles, regions, sheet_paths, background_crop):
    """
    Draws every combination, labeled, on sheets of at most MAX_COMBINATIONS_PER_SHEET; existing files are replaced.

    Args:
        crop_tiles: crop_tiles[i][j] is the RGBA image of tile j of crop i.
        regions: crop_regions output.
        sheet_paths: one path per sheet (see save_outputs); their folders are created if needed.
        background_crop: index of the background crop, or None (it is left out of the labels).
    """
    image_width, image_height = IMAGE_WIDTH, IMAGE_HEIGHT
    combinations = tile_combinations(crop_tiles)
    sheets = [combinations[start:start + MAX_COMBINATIONS_PER_SHEET]
              for start in range(0, len(combinations), MAX_COMBINATIONS_PER_SHEET)]
    columns = math.ceil(math.sqrt(len(sheets[0])))
    font = ImageFont.load_default(size=SHEET_LABEL_FONT_SIZE)

    for sheet_combinations, sheet_path in zip(sheets, sheet_paths, strict=True):
        rows = math.ceil(len(sheet_combinations) / columns)
        cell_width, cell_height = image_width + SHEET_GAP_PIXELS, image_height + SHEET_LABEL_HEIGHT + SHEET_GAP_PIXELS
        sheet = Image.new("RGB", (columns * cell_width + SHEET_GAP_PIXELS, rows * cell_height + SHEET_GAP_PIXELS), "white")
        draw = ImageDraw.Draw(sheet)
        for position, combination in enumerate(sheet_combinations):
            left = SHEET_GAP_PIXELS + (position % columns) * cell_width
            top = SHEET_GAP_PIXELS + (position // columns) * cell_height
            sheet.paste(assemble_combination(crop_tiles, regions, combination, (image_width, image_height)), (left, top))
            draw.text((left, top + image_height + 4), combination_label(combination, crop_tiles, background_crop),
                      fill="black", font=font)
        sheet_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(sheet, sheet_path)
        print(f"Saved sheet: {sheet_path}")


def save_separate_tiles(crop_tiles, crop_map, tiles_per_crop, background_crop, tiles_folder):
    """
    Saves every tile as its own image, one subfolder per non-empty crop, plus layout.json for the viewer.
    The background crop has a single tile, in cropBg/.

    Args:
        crop_tiles: crop_tiles[i][j] is the RGBA image of tile j of crop i.
        crop_map: [patch rows, patch cols] array of crop indices.
        tiles_per_crop: k.
        background_crop: index of the background crop, or None.
        tiles_folder: Path of the folder to write (created if needed).
    """
    tiles_folder.mkdir(parents=True, exist_ok=True)
    for crop_index, tiles in enumerate(crop_tiles):
        for tile_index, tile in enumerate(tiles):
            crop_folder = tiles_folder / crop_folder_name(crop_index, background_crop)
            crop_folder.mkdir(exist_ok=True)
            save_image(tile, crop_folder / f"tile{tile_index}.png")
    layout = {"width": IMAGE_WIDTH, "height": IMAGE_HEIGHT, "tiles_per_crop": tiles_per_crop,
              "num_crops": len(crop_tiles), "background_crop": background_crop,
              "crop_map": np.asarray(crop_map).tolist()}
    (tiles_folder / LAYOUT_FILENAME).write_text(json.dumps(layout))


def save_outputs(output, config, output_root):
    """
    Saves a pipeline result as the config asks (sheets, separate tiles, saliency maps) and prints where.
    The folder layout is described at the top of this file.

    Args:
        output: MixNMatchOutput.
        config: MixNMatchConfig.
        output_root: folder to write into: OUTPUT_DIR, or OUTPUT_DIR/<config folder name> for a folder run.

    Returns:
        Path of the tiles folder, or None if save_separate_tiles is false.
    """
    tile_images = [to_pil_image(image) for image in output.tiles]
    # Dynamic cropping adds the background crop, the last one; static cropping has none.
    background_crop = None if config.static_cropping else config.num_crops
    num_crops = config.num_crops if config.static_cropping else config.num_crops + 1
    regions = crop_regions(output.crop_map, num_crops)
    # The background crop is the same in every image (the model keeps its patches once), so it gets one tile.
    crop_tiles = [[] if region is None else
                  [cut_tile(image, region) for image in (tile_images[:1] if crop_index == background_crop
                                                         else tile_images)]
                  for crop_index, region in enumerate(regions)]

    run_folder = output_root / f"{config.prefix}_{config.num_crops}crops_{config.tiles_per_crop}tiles"
    if config.save_saliency_maps and not config.static_cropping:
        saliency_folder = run_folder / SALIENCY_DIRNAME
        save_saliency_outputs(output.saliency_maps, output.saliency_crop_maps, output.crop_map,
                              output.saliency_image, config.tiles_per_crop, saliency_folder)
        print(f"Saved saliency maps: {saliency_folder}")
    if config.visualize:
        num_sheets = math.ceil(len(tile_combinations(crop_tiles)) / MAX_COMBINATIONS_PER_SHEET)
        sheet_paths = [run_folder / SINGLE_SHEET_FILENAME] if num_sheets == 1 else [
            run_folder / SHEETS_DIRNAME / f"part{part}of{num_sheets}.png" for part in range(1, num_sheets + 1)
        ]
        save_visualization_sheets(crop_tiles, regions, sheet_paths, background_crop)
    if not config.save_separate_tiles:
        return None
    tiles_folder = run_folder / TILES_DIRNAME
    save_separate_tiles(crop_tiles, output.crop_map, config.tiles_per_crop, background_crop, tiles_folder)
    print(f"Saved tiles: {tiles_folder}")
    return tiles_folder


def save_standard_run(output, config, output_root):
    """
    Saves the image of a standard run as <output_root>/<prefix>_std_run.png (replacing an existing one) and prints
    where.

    Args:
        output: MixNMatchOutput of a standard run (tiles [1, 3, H, W] in [0, 1]).
        config: MixNMatchConfig.
        output_root: folder to write into (see save_outputs).
    """
    output_root.mkdir(parents=True, exist_ok=True)
    image_path = output_root / f"{config.prefix}_std_run.png"
    save_image(to_pil_image(output.tiles[0]), image_path)
    print(f"Saved image: {image_path}")


class CombinationViewer:
    """Window showing one combination; clicks change the tile of the clicked crop and Ctrl+S saves the image."""

    def __init__(self, window, tiles_folder):
        """
        Loads a tiles folder and builds the window contents.

        Args:
            window: tk.Tk root window.
            tiles_folder: Path of a folder written by save_separate_tiles.

        Raises:
            FileNotFoundError, KeyError, json.JSONDecodeError: if the folder is not a complete tiles folder.
        """
        layout = json.loads((tiles_folder / LAYOUT_FILENAME).read_text())
        self.tiles_folder = tiles_folder
        self.crop_map = np.asarray(layout["crop_map"])
        self.regions = crop_regions(self.crop_map, layout["num_crops"])
        self.image_size = (layout["width"], layout["height"])
        self.background_crop = layout["background_crop"]
        self.crop_tiles = [
            [] if region is None else
            [Image.open(tiles_folder / crop_folder_name(crop_index, self.background_crop) /
                        f"tile{tile_index}.png").convert("RGBA")
             for tile_index in range(1 if crop_index == self.background_crop else layout["tiles_per_crop"])]
            for crop_index, region in enumerate(self.regions)
        ]
        self.combination = [0] * len(self.regions)
        self.display_scale = min(1.0, MAX_DISPLAY_SIZE / max(self.image_size))
        self.display_size = tuple(round(side * self.display_scale) for side in self.image_size)

        window.title(f"Mix-And-Match viewer: {tiles_folder.resolve().parent.name}")  # the run folder's name
        self.canvas = tk.Canvas(window, width=self.display_size[0], height=self.display_size[1], highlightthickness=0)
        self.canvas.pack()
        self.status = tk.Label(window, anchor="w", justify="left")
        self.status.pack(fill="x", padx=6, pady=4)
        self.canvas.bind("<Button-1>", lambda event: self.change_tile(event, step=1))
        self.canvas.bind("<Button-3>", lambda event: self.change_tile(event, step=-1))
        window.bind("<Control-s>", self.save_combination)
        self.redraw()

    def current_image(self):
        """
        Returns:
            Full-resolution PIL image of the shown combination.
        """
        return assemble_combination(self.crop_tiles, self.regions, self.combination, self.image_size)

    def redraw(self, message=""):
        """
        Shows the current combination and the status line.

        Args:
            message: optional extra text for the status line.
        """
        self.photo = ImageTk.PhotoImage(self.current_image().resize(self.display_size))  # kept alive for Tk
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        help_text = "Left click: next tile   Right click: previous tile   Ctrl+S: save"
        label = combination_label(self.combination, self.crop_tiles, self.background_crop)
        self.status.config(text=f"{label}\n{help_text}\n{message}".strip())

    def change_tile(self, event, step):
        """
        Moves the clicked crop to its next (step=1) or previous (step=-1) tile, wrapping around.

        Args:
            event: Tk mouse event (display coordinates).
            step: +1 or -1.
        """
        row = min(int(event.y / self.display_scale) // PATCH_SIZE_PIXELS, self.crop_map.shape[0] - 1)
        col = min(int(event.x / self.display_scale) // PATCH_SIZE_PIXELS, self.crop_map.shape[1] - 1)
        crop_index = self.crop_map[row, col]
        self.combination[crop_index] = (self.combination[crop_index] + step) % len(self.crop_tiles[crop_index])
        self.redraw()

    def save_combination(self, event=None):
        """
        Saves the shown combination at full resolution to <tiles folder>/saved_combinations/.

        Args:
            event: Tk key event (unused).
        """
        save_folder = self.tiles_folder / SAVED_COMBINATIONS_DIRNAME
        save_folder.mkdir(exist_ok=True)
        tile_indices = "_".join(f"c{crop_index}t{tile_index}"
                                for crop_index, (tile_index, tiles) in enumerate(zip(self.combination, self.crop_tiles))
                                if tiles and crop_index != self.background_crop)
        save_path = save_folder / f"combination_{tile_indices}.png"
        save_image(self.current_image(), save_path)
        self.redraw(message=f"Saved {save_path}")


def run_viewer(tiles_folder):
    """
    Opens the viewer window for a tiles folder and blocks until it is closed.

    Args:
        tiles_folder: path of a folder written by save_separate_tiles.

    Raises:
        SystemExit: with a readable message if the folder is missing or incomplete.
    """
    tiles_folder = Path(tiles_folder)
    window = tk.Tk()
    try:
        CombinationViewer(window, tiles_folder)
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as error:
        window.destroy()
        sys.exit(f"Error: {tiles_folder} is not a complete Mix-And-Match tiles folder ({error})")
    window.mainloop()


def main():
    """Command-line entry: python vis_app.py <tiles folder>."""
    parser = argparse.ArgumentParser(description="Browse tile combinations of a Mix-And-Match tiles folder.")
    parser.add_argument("tiles_folder", help="folder written with save_separate_tiles (outputs/<run folder>/tiles)")
    run_viewer(parser.parse_args().tiles_folder)


if __name__ == "__main__":
    main()
