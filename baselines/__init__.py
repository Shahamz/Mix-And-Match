"""
The baselines Mix-And-Match is compared against.

    python -m baselines.mnm_baseline <config or folder>                  one plain image per tile prompt
    python -m baselines.tiled_diffusion.run <config>                     tiled-diffusion, on PixelDiT
    python -m baselines.regional_prompting.infer_pixeldit_regional <config>   Regional Prompting, on PixelDiT

Every baseline reads the same config files as generate.py, uses the same PixelDiT checkpoint, prompt encoding and
prompt cache, and writes into outputs/<baseline name>/.
"""
