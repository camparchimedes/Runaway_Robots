# TODO: review, and review again.
# TODO: these weights are generic, mod them for an more interesting turn @def load_pipe_into_UNet(myUNet, pipe_unet):
# TODO:Piping an XL model (XL might needs mods to attention, definitely other weights!)
"""
!pip install einops
!pip install diffusers transformers tokenizers
!pip install - -upgrade diffusers
!pip install accelerate
print(torch.__version__)
"""

# Import necessary libraries
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
import sys
import random
import mediapy as media
import matplotlib.pyplot as plt
from torch import autocast
from easydict import EasyDict as edict
from collections import OrderedDict
from einops import rearrange
import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import os
from PIL import Image


# Define Unet Architecture
# Residual Block /backbone
class ResBlock(nn.Module):
    def __init__(self, in_channel, time_emb_dim, out_channel=None):
        super().__init__()
        if out_channel is None:
            out_channel = in_channel
        self.norm1 = nn.GroupNorm(32, in_channel, eps=1e-05, affine=True)
        self.conv1 = nn.Conv2d(in_channel, out_channel,
                               kernel_size=3, stride=1, padding=1)
        self.time_emb_proj = nn.Linear(
            in_features=time_emb_dim, out_features=out_channel, bias=True)
        self.norm2 = nn.GroupNorm(32, out_channel, eps=1e-05, affine=True)
        self.dropout = nn.Dropout(p=0.0, inplace=False)
        self.conv2 = nn.Conv2d(out_channel, out_channel,
                               kernel_size=3, stride=1, padding=1)
        self.nonlinearity = nn.SiLU()
        if in_channel == out_channel:
            self.conv_shortcut = nn.Identity()
        else:
            self.conv_shortcut = nn.Conv2d(
                in_channel, out_channel, kernel_size=1, stride=1)

    def forward(self, x, t_emb, cond=None):
        h = self.norm1(x)
        h = self.nonlinearity(h)
        h = self.conv1(h)
        if t_emb is not None:
            t_hidden = self.time_emb_proj(self.nonlinearity(t_emb))
            h = h + t_hidden[:, :, None, None]
        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.conv_shortcut(x)


# UpSampling
class UpSample(nn.Module):
    def __init__(self, channel, scale_factor=2, mode='nearest'):
        super(UpSample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(
            channel, channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return self.conv(x)


# DownSampling
class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(
            channel, channel, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # F.interpolate(x, scale_factor=1/self.scale_factor, mode=self.mode)
        return self.conv(x)


# Attempt to use VAE to affect attention weights/ make the attention computation probability-hinged
class AttentionVAE(nn.Module):
    def __init__(self, embed_dim, hidden_dim, latent_dim, num_heads=8):
        super(AttentionVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim * num_heads)
        )
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        # Flatten
        x_flat = x.view(batch_size * seq_len, -1)

        encoded = self.encoder(x_flat)
        mu, log_var = encoded.chunk(2, dim=-1)
        z = self.reparameterize(mu, log_var)

        vae_weights = self.decoder(z)
        # Reshape /match expected output shape
        vae_weights = vae_weights.view(
            batch_size, seq_len, self.num_heads, self.embed_dim)
        return vae_weights, mu, log_var

    def vae_loss(mu, logvar):
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kld_loss

        vae_loss = vae_loss(mu, logvar)


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, hidden_dim, context_dim=None, num_heads=8, latent_dim=128):
        super(CrossAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.to_q = nn.Linear(hidden_dim, embed_dim, bias=False)
        # <-- ok..
        self.attention_vae = AttentionVAE(
            embed_dim=embed_dim,  hidden_dim=hidden_dim, latent_dim=latent_dim, num_heads=num_heads)
        if context_dim is None:
            # Self Attention
            self.to_k = nn.Linear(hidden_dim, embed_dim, bias=False)
            self.to_v = nn.Linear(hidden_dim, embed_dim, bias=False)
            self.self_attn = True
        else:
            # Cross Attention
            self.to_k = nn.Linear(context_dim, embed_dim, bias=False)
            self.to_v = nn.Linear(context_dim, embed_dim, bias=False)
            self.self_attn = False
        self.to_out = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, bias=True))

    def forward(self, tokens, context=None):
        Q = self.to_q(tokens)
        K = self.to_k(tokens) if self.self_attn else self.to_k(context)
        V = self.to_v(tokens) if self.self_attn else self.to_v(context)
        vae_weights, mu, logvar = self.attention_vae(tokens)

        print(Q.shape, K.shape, V.shape)

        Q = rearrange(Q, 'B T (H D) -> (B H) T D',
                      H=self.num_heads, D=self.head_dim)
        K = rearrange(K, 'B T (H D) -> (B H) T D',
                      H=self.num_heads, D=self.head_dim)
        V = rearrange(V, 'B T (H D) -> (B H) T D',
                      H=self.num_heads, D=self.head_dim)
        vae_weights = vae_weights.view(
            batch_size, self.num_heads, seq_len, self.head_dim)

        scoremats = torch.einsum("BHTD,BHSD->BHTS", Q, K) * vae_weights
        attnmats = F.softmax(scoremats / math.sqrt(self.head_dim), dim=-1)
        ctx_vecs = torch.einsum("BTS,BSD->BTD", attnmats, V)
        ctx_vecs = rearrange(ctx_vecs, '(B H) T D -> B T (H D)',
                             H=self.num_heads, D=self.head_dim)
        return self.to_out(ctx_vecs)

        class GEGLU_proj(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(GEGLU_proj, self).__init__()
        self.proj = nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x):
        x = self.proj(x)
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward_GEGLU(nn.Module):
    def __init__(self, hidden_dim, mult=4):
        super(FeedForward_GEGLU, self).__init__()
        self.net = nn.Sequential(
            GEGLU_proj(hidden_dim, mult * hidden_dim),
            nn.Dropout(0.0),
            nn.Linear(mult * hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)


# Transformer layers
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, context_dim, num_heads=8):
        super(TransformerBlock, self).__init__()
        self.attn1 = CrossAttention(
            hidden_dim, hidden_dim, num_heads=num_heads)  # self attention
        self.attn2 = CrossAttention(
            hidden_dim, hidden_dim, context_dim, num_heads=num_heads)  # cross attention
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ff = FeedForward_GEGLU(hidden_dim,)  # --->

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    def __init__(self, hidden_dim, context_dim, num_heads=8):
        super(SpatialTransformer, self).__init__()
        self.norm = nn.GroupNorm(32, hidden_dim, eps=1e-6, affine=True)
        self.proj_in = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.transformer = TransformerBlock(
            hidden_dim, context_dim, num_heads=8)  # <-------𝙬𝙝𝙖𝙩 𝙞𝙨?
        self.transformer_blocks = nn.Sequential(
            TransformerBlock(hidden_dim, context_dim, num_heads=8))
        self.proj_out = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x, cond=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.proj_in(self.norm(x))
        x = rearrange(x, "b c h w->b (h w) c")
        x = self.transformer_blocks[0](x, cond)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return self.proj_out(x) + x_in

"""
     self.ff = nn.Sequential(
         nn.Linear(hidden_dim, 3 * hidden_dim),
         nn.GELU(),
         nn.Linear(3 * hidden_dim, hidden_dim)
     )
     
    """

# Container of ResBlock and Spatial Transformers
# Modified Container.


class TimeModulatedSequential(nn.Sequential):
    def forward(self, x, t_emb, cond=None):
        for module in self:
            if isinstance(module, TimeModulatedSequential):
                x = module(x, t_emb, cond)
            elif isinstance(module, ResBlock):  # adding time modulation
                x = module(x, t_emb)
            elif isinstance(module, SpatialTransformer):  # adding class conditioning
                x = module(x, cond=cond)
            else:
                x = module(x)
        return x


# UNet | `class UNet_SD(nn.Module):`
class UNet_SD(nn.Module):
    def __init__(self, in_channels=4, base_channels=320, time_emb_dim=1280, context_dim=768, multipliers=(1, 2, 4, 4), attn_levels=(0, 1, 2), nResAttn_block=2, cat_unet=True):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.in_channels = in_channels
        self.out_channels = out_channels
        base_channels = base_channels
        time_emb_dim = time_emb_dim
        context_dim = context_dim
        multipliers = multipliers
        nlevel = len(multipliers)
        self.base_channels = base_channels
        attn_levels = [0, 1, 2]  # hmm..
        level_channels = [base_channels * mult for mult in multipliers]
        # Transform time into embedding
        self.time_embedding = nn.Sequential(OrderedDict({
            "linear_1": nn.Linear(base_channels, time_emb_dim, bias=True),
            "act": nn.SiLU(),  # <-------𝙨𝙬𝙞𝙩𝙘𝙝 𝙖𝙘𝙩𝙞𝙫𝙖𝙩𝙞𝙤𝙣?
            "linear_2": nn.Linear(time_emb_dim, time_emb_dim, bias=True),
        }))  # 2 layer MLP
        self.conv_in = nn.Conv2d(
            self.in_channels, base_channels, 3, stride=1, padding=1)
        nResAttn_block = nResAttn_block           # Tensor Downsample blocks
        self.down_blocks = TimeModulatedSequential()   # nn.ModuleList()
        self.down_blocks_channels = [base_channels]
        cur_chan = base_channels

        for i in range(nlevel):
            for j in range(nResAttn_block):
                res_attn_sandwich = TimeModulatedSequential()
                # input_chan of first ResBlock/ different
                res_attn_sandwich.append(ResBlock(
                    in_channel=cur_chan, time_emb_dim=time_emb_dim, out_channel=level_channels[i]))

                if i in attn_levels:   # add attention except for the last level
                    res_attn_sandwich.append(SpatialTransformer(
                        level_channels[i], context_dim=context_dim))
                cur_chan = level_channels[i]
                self.down_blocks.append(res_attn_sandwich)
                self.down_blocks_channels.append(cur_chan)
                res_attn_sandwich.append(DownSample(level_channels[i]))

            if not i == nlevel - 1:
                self.down_blocks.append(TimeModulatedSequential(
                    DownSample(level_channels[i])))
                self.down_blocks_channels.append(cur_chan)

        self.mid_block = TimeModulatedSequential(
            ResBlock(cur_chan, time_emb_dim),
            SpatialTransformer(cur_chan, context_dim=context_dim),
            ResBlock(cur_chan, time_emb_dim),
        )

        # Tensor Upsample blocks
        self.up_blocks = nn.ModuleList()  # ref. TimeModulatedSequential()
        for i in reversed(range(nlevel)):
            for j in range(nResAttn_block + 1):
                res_attn_sandwich = TimeModulatedSequential()
                res_attn_sandwich.append(ResBlock(in_channel=cur_chan + self.down_blocks_channels.pop(
                ), time_emb_dim=time_emb_dim, out_channel=level_channels[i]))
                if i in attn_levels:
                    res_attn_sandwich.append(SpatialTransformer(
                        level_channels[i], context_dim=context_dim))
                cur_chan = level_channels[i]
                if j == nResAttn_block and i != 0:
                    res_attn_sandwich.append(UpSample(level_channels[i]))
                self.up_blocks.append(res_attn_sandwich)
        # Read out from tensor to latent space
        self.output = nn.Sequential(
            nn.GroupNorm(32, base_channels),
            # nn.SiLU(),
            nn.GELU(),  # <-------𝙨𝙬𝙞𝙩𝙘𝙝ed 𝙖𝙘𝙩𝙞𝙫𝙖𝙩𝙞𝙤𝙣
            nn.Conv2d(base_channels, self.out_channels, 3, padding=1),
        )
        self.to(self.device)

    def time_proj(self, time_steps, max_period: int = 10000):
        if time_steps.ndim == 0:
            time_steps = time_steps.unsqueeze(0)
        half = self.base_channels // 2
        frequencies = torch.exp(-math.log(max_period) * torch.arange(
            start=0, end=half, dtype=torch.float32) / half).to(device=time_steps.device)
        angles = time_steps[:, None].float() * frequencies[None, :]
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

    def forward(self, x, time_steps, cond=None, encoder_hidden_states=None, output_dict=True):
        if cond is None and encoder_hidden_states is not None:
            cond = encoder_hidden_states
        t_emb = self.time_proj(time_steps)
        t_emb = self.time_embedding(t_emb)
        x = self.conv_in(x)
        down_x_cache = [x]
        for module in self.down_blocks:
            x = module(x, t_emb, cond)
            down_x_cache.append(x)
        x = self.mid_block(x, t_emb, cond)
        for module in self.up_blocks:
            x = module(torch.cat((x, down_x_cache.pop()), dim=1), t_emb, cond)
        x = self.output(x)
        if output_dict:
            return edict(sample=x)
        else:
            return x

pipe = AutoPipelineForText2Image.from_pretrained(
    'lykon/dreamshaper-xl-lightning', torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Safety dummy
def dummy_checker(images, **kwargs): return images, False
pipe.safety_checker = dummy_checker

# Standard v1.5 (see: TODO)
def load_pipe_into_UNet(myUNet, pipe_unet):
    # load the pretrained weights from the pipe into UNet.
    # Loading input and output layers.
    myUNet.output[0].load_state_dict(pipe_unet.conv_norm_out.state_dict())
    myUNet.output[2].load_state_dict(pipe_unet.conv_out.state_dict())
    myUNet.conv_in.load_state_dict(pipe_unet.conv_in.state_dict())
    myUNet.time_embedding.load_state_dict(
        pipe_unet.time_embedding.state_dict())
    # # Loading the down blocks
    myUNet.down_blocks[0][0].load_state_dict(
        pipe_unet.down_blocks[0].resnets[0].state_dict())
    myUNet.down_blocks[0][1].load_state_dict(
        pipe_unet.down_blocks[0].attentions[0].state_dict())
    myUNet.down_blocks[1][0].load_state_dict(
        pipe_unet.down_blocks[0].resnets[1].state_dict())
    myUNet.down_blocks[1][1].load_state_dict(
        pipe_unet.down_blocks[0].attentions[1].state_dict())
    myUNet.down_blocks[2][0].load_state_dict(
        pipe_unet.down_blocks[0].downsamplers[0].state_dict())

    myUNet.down_blocks[3][0].load_state_dict(
        pipe_unet.down_blocks[1].resnets[0].state_dict())
    myUNet.down_blocks[3][1].load_state_dict(
        pipe_unet.down_blocks[1].attentions[0].state_dict())
    myUNet.down_blocks[4][0].load_state_dict(
        pipe_unet.down_blocks[1].resnets[1].state_dict())
    myUNet.down_blocks[4][1].load_state_dict(
        pipe_unet.down_blocks[1].attentions[1].state_dict())
    myUNet.down_blocks[5][0].load_state_dict(
        pipe_unet.down_blocks[1].downsamplers[0].state_dict())

    myUNet.down_blocks[6][0].load_state_dict(
        pipe_unet.down_blocks[2].resnets[0].state_dict())
    myUNet.down_blocks[6][1].load_state_dict(
        pipe_unet.down_blocks[2].attentions[0].state_dict())
    myUNet.down_blocks[7][0].load_state_dict(
        pipe_unet.down_blocks[2].resnets[1].state_dict())
    myUNet.down_blocks[7][1].load_state_dict(
        pipe_unet.down_blocks[2].attentions[1].state_dict())
    myUNet.down_blocks[8][0].load_state_dict(
        pipe_unet.down_blocks[2].downsamplers[0].state_dict())

    myUNet.down_blocks[9][0].load_state_dict(
        pipe_unet.down_blocks[3].resnets[0].state_dict())
    myUNet.down_blocks[10][0].load_state_dict(
        pipe_unet.down_blocks[3].resnets[1].state_dict())

    # # Loading the middle blocks
    myUNet.mid_block[0].load_state_dict(
        pipe_unet.mid_block.resnets[0].state_dict())
    myUNet.mid_block[1].load_state_dict(
        pipe_unet.mid_block.attentions[0].state_dict())
    myUNet.mid_block[2].load_state_dict(
        pipe_unet.mid_block.resnets[1].state_dict())
    # Loading the up blocks /
    # upblock 0
    myUNet.up_blocks[0][0].load_state_dict(
        pipe_unet.up_blocks[0].resnets[0].state_dict())
    myUNet.up_blocks[1][0].load_state_dict(
        pipe_unet.up_blocks[0].resnets[1].state_dict())
    myUNet.up_blocks[2][0].load_state_dict(
        pipe_unet.up_blocks[0].resnets[2].state_dict())
    myUNet.up_blocks[2][1].load_state_dict(
        pipe_unet.up_blocks[0].upsamplers[0].state_dict())
    # # upblock 1
    myUNet.up_blocks[3][0].load_state_dict(
        pipe_unet.up_blocks[1].resnets[0].state_dict())
    myUNet.up_blocks[3][1].load_state_dict(
        pipe_unet.up_blocks[1].attentions[0].state_dict())
    myUNet.up_blocks[4][0].load_state_dict(
        pipe_unet.up_blocks[1].resnets[1].state_dict())
    myUNet.up_blocks[4][1].load_state_dict(
        pipe_unet.up_blocks[1].attentions[1].state_dict())
    myUNet.up_blocks[5][0].load_state_dict(
        pipe_unet.up_blocks[1].resnets[2].state_dict())
    myUNet.up_blocks[5][1].load_state_dict(
        pipe_unet.up_blocks[1].attentions[2].state_dict())
    myUNet.up_blocks[5][2].load_state_dict(
        pipe_unet.up_blocks[1].upsamplers[0].state_dict())
    # # upblock 2
    myUNet.up_blocks[6][0].load_state_dict(
        pipe_unet.up_blocks[2].resnets[0].state_dict())
    myUNet.up_blocks[6][1].load_state_dict(
        pipe_unet.up_blocks[2].attentions[0].state_dict())
    myUNet.up_blocks[7][0].load_state_dict(
        pipe_unet.up_blocks[2].resnets[1].state_dict())
    myUNet.up_blocks[7][1].load_state_dict(
        pipe_unet.up_blocks[2].attentions[1].state_dict())
    myUNet.up_blocks[8][0].load_state_dict(
        pipe_unet.up_blocks[2].resnets[2].state_dict())
    myUNet.up_blocks[8][1].load_state_dict(
        pipe_unet.up_blocks[2].attentions[2].state_dict())
    myUNet.up_blocks[8][2].load_state_dict(
        pipe_unet.up_blocks[2].upsamplers[0].state_dict())
    # # upblock 3
    myUNet.up_blocks[9][0].load_state_dict(
        pipe_unet.up_blocks[3].resnets[0].state_dict())
    myUNet.up_blocks[9][1].load_state_dict(
        pipe_unet.up_blocks[3].attentions[0].state_dict())
    myUNet.up_blocks[10][0].load_state_dict(
        pipe_unet.up_blocks[3].resnets[1].state_dict())
    myUNet.up_blocks[10][1].load_state_dict(
        pipe_unet.up_blocks[3].attentions[1].state_dict())
    myUNet.up_blocks[11][0].load_state_dict(
        pipe_unet.up_blocks[3].resnets[2].state_dict())
    myUNet.up_blocks[11][1].load_state_dict(
        pipe_unet.up_blocks[3].attentions[2].state_dict())


myunet = UNet_SD()
original_unet = pipe.unet.cpu()
load_pipe_into_UNet(myunet, original_unet)

pipe.unet = myunet.cuda()


# Begin prelim_test with the AttentionVAE, using the 
prompt = "masterly piece, load the pretrained weights from the pipe into my UNet, CG Art"
with autocast("cuda"):
    output = pipe(prompt)

image = output.images[0]  # Accessing the first image in the 'images' list
save_path = "/content/drive/MyDrive/vae_test_runs"
filename = f"generated_image_{len(os.listdir(save_path))}.jpg"
image_path = os.path.join(save_path, filename)
image.save(image_path)

print(f"Generated image saved at: {image_path}")
