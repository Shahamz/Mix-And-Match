MIX-AND-MATCH - QUICK START
===========================

Requirements: a CUDA GPU (8 GB is enough with "sequential_cfg": true) and Python 3.10+.

1. Install
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   (pick the build for your CUDA)
    pip install -r requirements.txt
    sudo apt install python3-tk                                                          (only for --vis_app)

2. Run
    python generate.py configs/template.json

   The first run downloads PixelDiT (nvidia/PixelDiT-1300M-1024px) and its Gemma text encoder
   (Efficient-Large-Model/gemma-2-2b-it) into the Hugging Face cache. To use weights you already have instead:
    python generate.py configs/template.json --weights-dir <folder>
   where <folder> holds config.json, pixeldit_t2i_v1.pth and gemma-2-2b-it/.

   Results go to outputs/<prefix>_<num_crops>crops_<tiles_per_crop>tiles/ (tiles/, visSheet.png, saliency/).

3. More
    python generate.py configs/template.json --vis_app   then browse the tile combinations (click a crop to change
                                                         its tile, Ctrl+S saves the shown image)
    python vis_app.py <run folder>/tiles                 browse an earlier result
    python generate.py configs/                          run every .json in a folder (results in outputs/configs/)
    python generate.py configs/template.json --std       plain PixelDiT with the background prompt only

   To make your own run, copy configs/template.json and edit the prompts, num_crops, tiles_per_crop, seed,
   cfg_scale and late_prompting. If the GPU runs out of memory, set "sequential_cfg": true or use fewer tiles.
