"""
The Regional Prompting baseline (https://github.com/instantX-research/Regional-Prompting-FLUX), ported to PixelDiT.

Every region gets its own prompt, the region prompts and the image tokens are masked against each other in the
joint attention, an unmasked "base" stream runs alongside and is blended into the image tokens, and all of it only
for the first few denoising steps. Run it with:

    python -m baselines.regional_prompting.infer_pixeldit_regional <config>
"""
