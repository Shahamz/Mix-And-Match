"""
PixelDiT under Regional-Prompting control: the project's model with a regional patch stage.

Only the patch stage changes, so everything else is inherited from pixeldit_model.py - the layers, the weights and
their names, the pixel stage and the loading. What the three classes below add mirrors
Regional-Prompting-FLUX's transformer_flux.py:

    * text_embeds is [batch, P, prompt length, text dim] - one prompt block per region. The model already gives
      every prompt the same learned positions and the same 1D text RoPE, which is the analogue of FLUX zeroing
      txt_ids for all regional prompts (transformer_flux.py:538). Regions are located purely by the attention
      mask, never by position.
    * regional_attention_mask masks the joint [text | image] attention of the patch blocks.
    * text_embeds_base / base_ratio run a SECOND, unmasked text stream over the SAME image tokens and blend the
      two image outputs after every attention (transformer_flux.py:287):
          x = x_regional * (1 - base_ratio) + x_base * base_ratio
    * inject_blocks_interval: a block is masked iff index % interval == 0; the other blocks still run the dual
      stream, unmasked (transformer_flux.py:597).

A run here is always a single image, so none of the tile paths of pipeline_mix_n_match exist in this forward. The
two pixel-stage blocks carry no text and have no FLUX analogue, so they always run unmasked.
"""

import torch
import torch.nn as nn

from attention import attention
from pixeldit_model import MMDiTBlockT2I, MMDiTJointAttention, PixDiT_T2I, apply_adaln, apply_rotary_emb


class RegionalJointAttention(MMDiTJointAttention):
    """Joint attention that masks the region prompts and can run a second, unmasked base stream beside them."""

    def forward(self, x, y, y_base, pos_img, pos_txt, pos_txt_base, attention_mask, base_ratio):
        """
        Args:
            x: [batch, patches, dim] image tokens.
            y: [batch, P * prompt length, dim] text tokens, one block per region.
            y_base: [batch, prompt length, dim] base-prompt tokens, or None without a base stream.
            pos_img: [patches, head dim // 2] image RoPE frequencies.
            pos_txt: [P * prompt length, head dim // 2] text RoPE frequencies.
            pos_txt_base: [prompt length, head dim // 2] base text RoPE frequencies, or None.
            attention_mask: mask over [text | image], or None for full attention.
            base_ratio: weight of the base stream's image output, or None to run one stream only.

        Returns:
            (image output, text output, base text output); the last is None without a base stream.
        """
        B, Nx, C = x.shape
        qkv_x = self.qkv_x(x).reshape(B, Nx, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        qx, kx, vx = self.q_norm_x(qkv_x[0]), self.k_norm_x(qkv_x[1]), qkv_x[2]
        qx, kx = apply_rotary_emb(qx, kx, freqs_cis=pos_img)

        out_x, out_y = self.joint_attention(qx, kx, vx, y, pos_txt, attention_mask)
        if base_ratio is None:
            return out_x, out_y, None

        # The base stream sees the SAME image tokens with the base prompt and no mask; only the image half of
        # the two outputs is blended (transformer_flux.py:287).
        out_x_base, out_y_base = self.joint_attention(qx, kx, vx, y_base, pos_txt_base, None)
        out_x = out_x * (1 - base_ratio) + out_x_base * base_ratio
        return out_x, out_y, out_y_base

    def joint_attention(self, qx, kx, vx, y, pos_txt, attention_mask):
        """
        One joint attention pass of text tokens against image tokens that are already projected.

        Args:
            qx, kx, vx: [batch, patches, heads, head dim] image queries, keys and values (RoPE applied).
            y: [batch, text length, dim] text tokens.
            pos_txt: [text length, head dim // 2] text RoPE frequencies.
            attention_mask: mask over [text | image], or None.

        Returns:
            (image output, text output).
        """
        B, Ny, C = y.shape
        Nx = qx.shape[1]
        qkv_y = self.qkv_y(y).reshape(B, Ny, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        qy, ky, vy = self.q_norm_y(qkv_y[0]), self.k_norm_y(qkv_y[1]), qkv_y[2]
        qy, ky = apply_rotary_emb(qy, ky, freqs_cis=pos_txt)

        # Token order is [text, image], which is the order the regional attention mask is built in.
        q_joint = torch.cat([qy, qx], dim=1)
        k_joint = torch.cat([ky, kx], dim=1)
        v_joint = torch.cat([vy, vx], dim=1)
        out_joint = attention(q_joint.transpose(1, 2), k_joint.transpose(1, 2), v_joint.transpose(1, 2),
                              attention_mask)

        out_joint = out_joint.transpose(1, 2)
        out_y = out_joint[:, :Ny].reshape(B, Ny, C)
        out_x = out_joint[:, Ny:].reshape(B, Nx, C)
        return self.proj_x(out_x), self.proj_y(out_y)


class RegionalMMDiTBlock(MMDiTBlockT2I):
    """Patch-stage block whose attention is the regional one, and which carries the base text stream along."""

    def __init__(self, hidden_size, groups):
        """
        Args:
            hidden_size: token size.
            groups: number of attention heads.
        """
        super().__init__(hidden_size, groups)
        self.attn = RegionalJointAttention(hidden_size, num_heads=groups, qkv_bias=False)

    def forward(self, x, y, y_base, c, pos_img, pos_txt, pos_txt_base, attention_mask, base_ratio):
        """
        Args:
            x: [batch, patches, hidden] image tokens.
            y: [batch, P * prompt length, hidden] regional text tokens.
            y_base: [batch, prompt length, hidden] base-prompt tokens, or None without a base stream.
            c: [batch, 1, hidden] timestep condition.
            pos_img, pos_txt, pos_txt_base: RoPE frequencies of the image, regional text and base text.
            attention_mask: mask over [text | image], or None.
            base_ratio: weight of the base stream, or None.

        Returns:
            (updated x, updated y, updated y_base).
        """
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = self.adaLN_modulation_img(c).chunk(6, dim=-1)
        shift_msa_y, scale_msa_y, gate_msa_y, shift_mlp_y, scale_mlp_y, gate_mlp_y = self.adaLN_modulation_txt(c).chunk(6, dim=-1)

        x_norm = apply_adaln(self.norm_x1(x), shift_msa_x, scale_msa_x)
        y_norm = apply_adaln(self.norm_y1(y), shift_msa_y, scale_msa_y)
        # The base text stream reuses the same text weights and the same modulation, the way FLUX's base stream
        # reuses norm1_context / ff_context (transformer_flux.py:208-265).
        y_base_norm = apply_adaln(self.norm_y1(y_base), shift_msa_y, scale_msa_y) if base_ratio is not None else None
        attn_x, attn_y, attn_y_base = self.attn(x_norm, y_norm, y_base_norm, pos_img, pos_txt, pos_txt_base,
                                                attention_mask, base_ratio)
        x = x + gate_msa_x * attn_x
        y = y + gate_msa_y * attn_y

        x = x + gate_mlp_x * self.mlp_x(apply_adaln(self.norm_x2(x), shift_mlp_x, scale_mlp_x))
        y = y + gate_mlp_y * self.mlp_y(apply_adaln(self.norm_y2(y), shift_mlp_y, scale_mlp_y))
        if base_ratio is not None:
            y_base = y_base + gate_msa_y * attn_y_base
            y_base = y_base + gate_mlp_y * self.mlp_y(apply_adaln(self.norm_y2(y_base), shift_mlp_y, scale_mlp_y))
        return x, y, y_base


class RegionalPixDiT_T2I(PixDiT_T2I):
    """PixelDiT whose patch blocks are the regional ones; the checkpoint loads into it unchanged."""

    def __init__(self, **architecture):
        """
        Args:
            **architecture: the checkpoint's architecture values, as PixDiT_T2I takes them.
        """
        super().__init__(**architecture)
        self.patch_blocks = nn.ModuleList([RegionalMMDiTBlock(self.hidden_size, self.num_groups)
                                           for _ in self.patch_blocks])

    def forward(self, images, timestep, text_embeds, text_embeds_base=None, base_ratio=None,
                regional_attention_mask=None, inject_blocks_interval=1):
        """
        Args:
            images: [batch, 3, H, W] noisy images in [-1, 1] scale.
            timestep: [batch] timesteps in [0, 1000].
            text_embeds: [batch, P, prompt length, text dim] prompt features. With regional control this is one
                prompt per region, in the order the attention mask was built in; otherwise a single prompt.
            text_embeds_base: [batch, 1, prompt length, text dim] whole-image prompt of the base stream, or None
                to run a single stream.
            base_ratio: weight of the base stream's image output, or None to run a single stream.
            regional_attention_mask: mask over [P * prompt length | patches], or None for full attention.
            inject_blocks_interval: a patch block is masked iff its index % this == 0; the others still run both
                streams, unmasked.

        Returns:
            [batch, 3, H, W] predicted flow velocity.
        """
        B, _, H, W = images.shape
        P = text_embeds.shape[1]
        Hs, Ws = H // self.patch_size, W // self.patch_size
        L = Hs * Ws

        x_patches = torch.nn.functional.unfold(images, kernel_size=self.patch_size, stride=self.patch_size)
        x_patches = x_patches.transpose(1, 2)
        pos = self.fetch_pos(Hs, Ws, images.device)

        t_emb = self.t_embedder(timestep.view(-1)).view(B, -1, self.hidden_size)
        condition = torch.nn.functional.silu(t_emb)

        # Every prompt gets its own copy of the learned positions and of the text RoPE, as if it were alone.
        # That is what lets one sequence hold P region prompts, and it is the analogue of FLUX giving every
        # regional prompt the position id (0, 0, 0) (transformer_flux.py:538).
        text_length = min(text_embeds.shape[2], self.txt_max_length)
        y_emb = self.y_embedder(text_embeds[:, :, :text_length]) + self.y_pos_embedding[:, :text_length].to(images.dtype)
        y_emb = y_emb.reshape(B, P * text_length, self.hidden_size)
        pos_txt_one = self.fetch_pos_text(text_length, images.device)
        pos_txt = pos_txt_one.repeat(P, 1)

        if base_ratio is not None:
            P_base = text_embeds_base.shape[1]
            y_base = (self.y_embedder(text_embeds_base[:, :, :text_length])
                      + self.y_pos_embedding[:, :text_length].to(images.dtype))
            y_base = y_base.reshape(B, P_base * text_length, self.hidden_size)
            pos_txt_base = pos_txt_one.repeat(P_base, 1)
        else:
            y_base, pos_txt_base = None, None

        s = self.s_embedder(x_patches)
        for index_block, block in enumerate(self.patch_blocks):
            # Blocks off the interval still run both streams, but see no mask.
            block_mask = regional_attention_mask if index_block % inject_blocks_interval == 0 else None
            s, y_emb, y_base = block(s, y_emb, y_base, condition, pos, pos_txt, pos_txt_base, block_mask,
                                     base_ratio)
        s = torch.nn.functional.silu(t_emb + s)

        s_cond = s.reshape(B * L, self.hidden_size)
        P2 = self.patch_size * self.patch_size
        x_pixels = self.pixel_embedder(images, img_height=H, img_width=W, patch_size=self.patch_size)
        x_pixels = x_pixels.view(B * L, P2, -1)
        for block in self.pixel_blocks:
            x_pixels = block(x_pixels, s_cond, H, W, self.patch_size)

        x_pixels = self.final_layer(x_pixels).view(B, L, P2, self.out_channels)
        x_pixels = x_pixels.permute(0, 3, 2, 1).reshape(B, self.out_channels * P2, L)
        return torch.nn.functional.fold(x_pixels, (H, W), kernel_size=self.patch_size, stride=self.patch_size)
