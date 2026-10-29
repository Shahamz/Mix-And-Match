"""
The tiled-diffusion baseline (https://github.com/madaror/tiled-diffusion), denoising with PixelDiT.

Its subject is the seam: tiles are generated together, and at every denoising step a band of each tile's padding
is copied from its neighbours in the crops' grid, so the candidates of one crop stay interchangeable at the join.
Run it with:

    python -m baselines.tiled_diffusion.run <config>
"""
