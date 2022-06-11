from functools import partial

import flax.linen as nn

from einops import rearrange, repeat
from grpc import Call

import jax.numpy as jnp
from jax.numpy import einsum

import numpy

from typing import Callable

MIN_DIM_HEAD = 32

def exists(x):
    return x is not None

def default(x, default_value):
    return x if exists(x) else default_value

def divisible_by(x, divisor):
    return (x / divisor).is_integer()

def cast_tuple(x, num = 1):
    return x if isinstance(x, tuple) else ((x,) * num)


class IdentityLayer(nn.Module):

  @nn.compact
  def __call__(self, x):
    return x

# normalization

class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-8
    gated: bool = False

    @nn.compact
    def __call__(self, x):
        scale = self.dim ** -0.5
        gamma = self.param('gamma', nn.initializers.ones, self.dim)
        weight = self.param('weight', nn.initializers.ones, self.dim) if self.gated else None
        norm = jnp.norm(x, axis = -1, keepdims = True) * scale
        out = (x / norm.clamp(min = self.eps)) * gamma
        if not exists(weight):
            return out
        return out * nn.sigmoid(x * weight)

# pre and post norm residual wrapper modules

class PreNorm(nn.Module):
    fn: Callable

    @nn.compact
    def __call__(self, x, **kwargs):
        x = nn.LayerNorm(epsilon = 1e-5, use_bias = False)(x)
        return self.fn(x, **kwargs)

class PostNorm(nn.Module): 
    dim: int 
    fn: Callable 
    scale_residual: int = 1 
    norm_klass = RMSNorm

    @nn.compact
    def forward(self, x, *args, **kwargs):
        fn = self.fn
        scale_residual = self.scale_residual
        norm = norm_klass(dim)
        residual = x * scale_residual
        out = fn(x, *args, **kwargs) + residual
        return norm(out)

# positional embedding

class RotaryEmbedding(nn.Module):
    dim: int

    @nn.compact
    def _call__(self, max_seq_len, *, offset = 0):
        inv_freq = 1. / (10000 ** (jnp.arange(0, self.dim, 2) / self.dim))
        seq = jnp.arange(max_seq_len) + offset
        freqs = einsum('i , j -> i j', seq.type_as(inv_freq), inv_freq)
        emb = jnp.concatenate((freqs, freqs), dim = -1)
        return rearrange(emb, 'n d -> 1 1 n d')

def jax_unstack(x, axis = 0):
    return jnp.moveaxis(x, axis, 0)

def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = jax_unstack(x, dim = -2)
    return jnp.concatenate((-x2, x1), dim = -1)

def apply_rotary_pos_emb(t, freqs):
    seq_len, rot_dim = t.shape[-2], freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * jnp.cos(freqs)) + (rotate_half(t) * jnp.sin(freqs))
    return jnp.concatenate((t, t_pass), dim = -1)

# feedforward

class FeedForward(nn.Module):
    dim: int
    mult: int = 4
    dropout: float = 0.

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features = int(self.dim * self.mult))(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate = self.dropout)(x, deterministic = False)
        x = nn.Dense(features = self.dim)(x)
        return x

# attention

class Attention(nn.Module):
    dim: int
    context_dim: None
    dim_heads: int = 64
    heads: int = 8
    causal: bool = False
    dropout: float = 0.
    null_kv: bool = False

    @nn.compact
    def __call__(self, x, mask = None, context = None, pos_emb = None):

        #context_dim = default(self.context_dim, self.dim)

        heads = self.heads
        scale = self.dim_heads ** -0.5
        causal = self.causal

        inner_dim = self.dim_heads * heads

        if self.null_kv:
            null_k = nn.Parameter(torch.randn(inner_dim))  
        else:
            None

        if self.null_kv:
            null_v = nn.Parameter(torch.randn(inner_dim))
        else:
            None

        b, h = x.shape[0], self.heads

        kv_input = default(context, x)

        q = nn.Dense(features = inner_dim, use_bias = False)(x)
        k = nn.Dense(features = inner_dim, use_bias = False)(kv_input)
        v = nn.Dense(features = inner_dim, use_bias = False)(kv_input)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # scale

        q = q * scale

        # apply relative positional encoding (rotary embeddings)

        if exists(pos_emb):
            q_pos_emb, k_pos_emb = cast_tuple(pos_emb, num = 2)

            q = apply_rotary_pos_emb(q, q_pos_emb)
            k = apply_rotary_pos_emb(k, k_pos_emb)

        # add null key / values

        if exists(null_k):
            nk, nv = null_k, null_v
            nk, nv = map(lambda t: repeat(t, '(h d) -> b h 1 d', b = b, h = h), (nk, nv))
            k = jnp.concatenate((nk, k), dim = -2)
            v = jnp.concatenate((nv, v), dim = -2)

        # derive query key similarities

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # masking

        mask_value = -jnp.finfo(sim.dtype).max

        if exists(mask):
            if exists(null_k):
                mask = F.pad(mask, (1, 0), value = True)

            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = jnp.where(~mask, sim, mask_value)

        if causal:
            i, j = sim.shape[-2:]
            causal_mask = jnp.ones(i, j).triu(j - i + 1)
            sim = jnp.where(causal_mask, sim, mask_value)

        # attention

        attn = sim.softmax(dim = -1)

        attn = nn.Dropout(rate = self.dropout)(attn, deterministic = False)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')

        # combine heads linear out
        to_out = nn.Dense(features = self.dim, use_bias = False)(out)

        return to_out


class ChunkedCrossAttention(nn.Module):
    chunk_size: int
    **kwargs

    @nn.compact
    def __call__(self, x, *, context_mask = None, context, pos_emb = None):
    
        cross_attn = Attention(null_kv = True, **kwargs)

        # derive variables
        chunk_size = self.chunk_size

        b, n, num_chunks, num_retrieved = x.shape[0], x.shape[-2], *context.shape[-4:-2]

        # if sequence length less than chunk size, do an early return

        if n < chunk_size:
            return jnp.zeros_like(x)

        # causal padding

        causal_padding = chunk_size - 1

        x = F.pad(x, (0, 0, -causal_padding, causal_padding), value = 0.)

        # remove sequence which is ahead of the neighbors retrieved (during inference)

        seq_index = (n // chunk_size) * chunk_size
        x, x_remainder = x[:, :seq_index], x[:, seq_index:]

        seq_remain_len = x_remainder.shape[-2]

        # take care of rotary positional embedding
        # make sure queries positions are properly shifted to the future

        q_pos_emb, k_pos_emb = pos_emb
        q_pos_emb = F.pad(q_pos_emb, (0, 0, -causal_padding, causal_padding), value = 0.)

        k_pos_emb = repeat(k_pos_emb, 'b h n d -> b h (r n) d', r = num_retrieved)
        pos_emb = (q_pos_emb, k_pos_emb)

        # reshape so we have chunk to chunk attention, without breaking causality

        x = rearrange(x, 'b (k n) d -> (b k) n d', k = num_chunks)
        context = rearrange(context, 'b k r n d -> (b k) (r n) d')

        if exists(context_mask):
            context_mask = rearrange(context_mask, 'b k r n -> (b k) (r n)')

        # cross attention

        out = cross_attn(x, context = context, mask = context_mask, pos_emb = pos_emb)

        # reshape back to original sequence

        out = rearrange(out, '(b k) n d -> b (k n) d', b = b)

        # pad back to original, with 0s at the beginning (which will be added to the residual and be fine)

        out = F.pad(out, (0, 0, causal_padding, -causal_padding + seq_remain_len), value = 0.)
        return out

# encoder and decoder classes

class Encoder(nn.Module):
    dim: int
    depth: int
    context_dim = None
    causal: bool = False
    heads: int = 8
    dim_head: int = 64
    attn_dropout: float = 0.
    ff_mult: int = 4
    ff_dropout: float = 0.
    final_norm: bool = True
    cross_attn_layers = None
    post_norm: bool = False
    output_dim = None
    norm_klass = RMSNorm
    scale_residual: float = 1.

    @nn.compact
    def __call__(self, x, *, mask = None, chunked_seq):

        layers = nn.ModuleList([])

        # partial rotary embeddings, which is better than full rotary
        # Wang and Komatsuzaki et al https://github.com/kingoflolz/mesh-transformer-jax/

        rotary_emb_dim = min(self.dim_head, MIN_DIM_HEAD)
        rotary_pos_emb = RotaryEmbedding(rotary_emb_dim)

        wrapper = partial(PreNorm, self.dim, norm_klass = norm_klass) if not self.post_norm else partial(PostNorm, dim, scale_residual = scale_residual, norm_klass = norm_klass)

        for layer_num in range(1, self.depth + 1):
            has_cross_attn = not exists(self.cross_attn_layers) or layer_num in self.cross_attn_layers

            layers.append(nn.ModuleList([
                wrapper(Attention(dim = self.dim, dim_head = self.dim_head, heads = self.heads, dropout = self.attn_dropout, causal = self.causal)),
                wrapper(Attention(dim = self.dim, context_dim = self.context_dim, dim_head = self.dim_head, heads = self.heads, dropout = self.attn_dropout)) if has_cross_attn else None,
                wrapper(FeedForward(dim = self.dim, mult = self.ff_mult, dropout = self.ff_dropout)),
            ]))

        self.norm_out = norm_klass(self.dim) if self.final_norm and not self.post_norm else nn.Identity()
        self.project_out = nn.Linear(dim, output_dim) if exists(output_dim) else nn.Identity()

        device, chunk_size, seq_len = x.device, x.shape[-2], chunked_seq.shape[-2]

        q_pos_emb = self.rotary_pos_emb(chunk_size, device = device)
        k_pos_emb = self.rotary_pos_emb(seq_len, device = device)

        for attn, cross_attn, ff in self.layers:
            x = attn(x, mask = mask, pos_emb = q_pos_emb)

            if exists(cross_attn):
                x = cross_attn(x, context = chunked_seq, pos_emb = (q_pos_emb, k_pos_emb))

            x = ff(x)

        x = self.norm_out(x)
        return self.project_out(x)
