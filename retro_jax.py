from functools import partial

import flax.linen as nn

from einops import rearrange, repeat
from grpc import Call

import jax.numpy as jnp
from jax.numpy import einsum

import numpy

from typing import Callable

# constants

MIN_DIM_HEAD = 32

BERT_VOCAB_SIZE = 28996

# helper functions

def exists(x):
    return x is not None

def default(x, default_value):
    return x if exists(x) else default_value

def divisible_by(x, divisor):
    return (x / divisor).is_integer()

def cast_tuple(x, num = 1):
    return x if isinstance(x, tuple) else ((x,) * num)

# Identity wrapper

class IdentityLayer(nn.Module):

  @nn.compact
  def __call__(self, x):
    return x

# deepnet init

def deepnorm_init(transformer, beta, module_name_match_list = ['.ff.', '.to_v', '.to_out']):
    for name, module in transformer.named_modules():
        if type(module) != nn.Linear:
            continue

        needs_beta_gain = any(map(lambda substr: substr in name, module_name_match_list))
        gain = beta if needs_beta_gain else 1
        nn.init.xavier_normal_(module.weight.data, gain = gain)

        if exists(module.bias):
            nn.init.constant_(module.bias.data, 0)

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
        norm = self.norm_klass(self.dim)
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

        layers = []

        # partial rotary embeddings, which is better than full rotary
        # Wang and Komatsuzaki et al https://github.com/kingoflolz/mesh-transformer-jax/

        rotary_emb_dim = min(self.dim_head, MIN_DIM_HEAD)
        rotary_pos_emb = RotaryEmbedding(rotary_emb_dim)

        wrapper = partial(PreNorm, self.dim, norm_klass = norm_klass) if not self.post_norm else partial(PostNorm, self.dim, scale_residual = self.scale_residual, norm_klass = norm_klass)

        for layer_num in range(1, self.depth + 1):
            has_cross_attn = not exists(self.cross_attn_layers) or layer_num in self.cross_attn_layers

            layers.append([
                wrapper(Attention(dim = self.dim, dim_head = self.dim_head, heads = self.heads, dropout = self.attn_dropout, causal = self.causal)),
                wrapper(Attention(dim = self.dim, context_dim = self.context_dim, dim_head = self.dim_head, heads = self.heads, dropout = self.attn_dropout)) if has_cross_attn else None,
                wrapper(FeedForward(dim = self.dim, mult = self.ff_mult, dropout = self.ff_dropout)),
            ])

        if self.final_norm and not self.post_norm:
            norm_out = norm_klass(self.dim)
        else: 
            IdentityLayer()
        
        project_out = nn.Dense(self.output_dim) if exists(self.output_dim) else IdentityLayer()

        chunk_size, seq_len = x.shape[-2], chunked_seq.shape[-2]

        q_pos_emb = rotary_pos_emb(chunk_size)
        k_pos_emb = rotary_pos_emb(seq_len)

        for attn, cross_attn, ff in self.layers:
            x = attn(x, mask = mask, pos_emb = q_pos_emb)

            if exists(cross_attn):
                x = cross_attn(x, context = chunked_seq, pos_emb = (q_pos_emb, k_pos_emb))

            x = ff(x)

        x = norm_out(x)
        return project_out(x)

class Decoder(nn.Module):
    dim: int
    depth: int
    heads = 8
    dim_head = 64
    attn_dropout = 0.
    ff_mult = 4
    ff_dropout = 0.
    final_norm = True
    cross_attn_layers = None
    chunk_size = 64
    post_norm = False
    norm_klass = RMSNorm
    scale_residual = 1.


    def __call__(self, x, *, encoder = None, encoder_retrieved_mask = None, context_mask = None, retrieved = None):
        layers = []

        # partial rotary embeddings, which is better than full rotary
        # Wang and Komatsuzaki et al https://github.com/kingoflolz/mesh-transformer-jax/

        rotary_emb_dim = min(self.dim_head, MIN_DIM_HEAD)
        rotary_pos_emb = RotaryEmbedding(rotary_emb_dim)

        wrapper = partial(PreNorm, self.dim, norm_klass = norm_klass) if not self.post_norm else partial(PostNorm, self.dim, scale_residual = self.scale_residual, norm_klass = norm_klass)

        chunk_size = self.chunk_size

        for layer_num in range(1, self.depth + 1):
            has_cross_attn = not exists(self.cross_attn_layers) or layer_num in self.cross_attn_layers

            self.layers.append([
                wrapper(Attention(dim = self.dim, dim_head = self.dim_head, heads = self.heads, dropout = self.attn_dropout, causal = True)),
                wrapper(ChunkedCrossAttention(chunk_size = chunk_size, dim = self.dim, dim_head = self.dim_head, heads = self.heads, dropout = self.attn_dropout)) if has_cross_attn else None,
                wrapper(FeedForward(dim = self.dim, mult = self.ff_mult, dropout = self.ff_dropout)),
            ])

        norm_out = norm_klass(self.dim) if self.final_norm and not self.post_norm else IdentityLayer()

        seq_len = x.shape[-2]
        self_attn_pos_emb = rotary_pos_emb(seq_len)

        # calculate seq index

        num_seq_chunks = seq_len // chunk_size
        seq_index = num_seq_chunks * chunk_size

        # rotary positions on the retrieved chunks

        if exists(retrieved):
            num_chunks, num_neighbors, chunk_size = retrieved.shape[-4:-1]

            cross_attn_q_pos_emb = rotary_pos_emb(chunk_size, offset = self.chunk_size - 1)  # need to add extra chunk size, since it will be shifted
            cross_attn_k_pos_emb = rotary_pos_emb(chunk_size)

            cross_attn_pos_emb = (cross_attn_q_pos_emb, cross_attn_k_pos_emb)

        # keep track of whether retrieved tokens are encoded yet

        retrieved_encoded = False

        # go through the decoder layers

        for attn, cross_attn, ff in layers:
            x = attn(x, pos_emb = self_attn_pos_emb)

            if exists(cross_attn) and exists(retrieved):
                if not retrieved_encoded:
                    retrieved = rearrange(retrieved, 'b k r n d -> (b k r) n d')
                    seq_as_context = repeat(x[:, :seq_index], 'b (k n) d -> (b k r) n d', n = chunk_size, r = num_neighbors)

                    retrieved = encoder(retrieved, mask = encoder_retrieved_mask, chunked_seq = seq_as_context)
                    retrieved = rearrange(retrieved, '(b k r) n d -> b k r n d', k = num_chunks, r = num_neighbors)
                    retrieved_encoded = True

                x = cross_attn(
                    x,
                    context = retrieved,
                    context_mask = context_mask,
                    pos_emb = cross_attn_pos_emb
                )

            x = ff(x)

        return norm_out(x)

# main class

class RETRO(nn.Module):
    num_tokens = BERT_VOCAB_SIZE,
    max_seq_len = 2048
    enc_dim = 896
    enc_depth = 2
    enc_cross_attn_layers = None
    dec_depth = 12
    dec_cross_attn_layers = (1, 3, 6, 9)
    heads = 8
    dec_dim = 768
    dim_head = 64
    enc_attn_dropout = 0.
    enc_ff_dropout = 0.
    dec_attn_dropout = 0.
    dec_ff_dropout = 0.
    chunk_size = 64
    pad_id = 0
    enc_scale_residual = None
    dec_scale_residual = None
    norm_klass = None
    gated_rmsnorm: bool = False
    use_deepnet: bool = False

    def __call__(self, seq, retrieved = None, return_loss = False):
        assert self.dim_head >= MIN_DIM_HEAD, f'dimension per head must be greater than {MIN_DIM_HEAD}'
        seq_len = self.max_seq_len
        pad_id = self.pad_id

        token_emb = nn.Embedding(self.num_tokens, self.enc_dim)
        pos_emb = nn.Embedding(self.max_seq_len, self.enc_dim)

        chunk_size = self.chunk_size

        if self.enc_dim != self.dec_dim:
            to_decoder_model_dim = nn.Dense(self.dec_dim)  
        else: 
            to_decoder_model_dim = IdentityLayer()

        # for deepnet, residual scales
        # follow equation in Figure 2. in https://arxiv.org/abs/2203.00555

        if self.use_deepnet:
            enc_scale_residual = default(enc_scale_residual, 0.81 * ((self.enc_depth ** 4) * self.dec_depth) ** .0625)
            dec_scale_residual = default(dec_scale_residual, (3 * self.dec_depth) ** 0.25)
            norm_klass = nn.LayerNorm()

        # allow for gated rmsnorm

        if self.gated_rmsnorm:
            norm_klass = partial(RMSNorm, gated = True)

        # define encoder and decoders

        encoder = Encoder(
            dim = self.enc_dim,
            context_dim = self.dec_dim,
            depth = self.enc_depth,
            attn_dropout = self.enc_attn_dropout,
            ff_dropout = self.enc_ff_dropout,
            cross_attn_layers = self.enc_cross_attn_layers,
            post_norm = self.use_deepnet,
            norm_klass = self.norm_klass,
            scale_residual = self.nc_scale_residual,
            output_dim = self.dec_dim
        )

        decoder = Decoder(
            dim = self.dec_dim,
            depth = self.dec_depth,
            attn_dropout = self.dec_attn_dropout,
            ff_dropout = self.ec_ff_dropout,
            cross_attn_layers = self.dec_cross_attn_layers,
            chunk_size = chunk_size,
            post_norm = self.use_deepnet,
            norm_klass = self.norm_klass,
            scale_residual = self.dec_scale_residual
        )

        to_logits = nn.Dense(self.num_tokens)

        # deepnet has special init of weight matrices

        if self.use_deepnet:
            deepnorm_init(self.encoder, 0.87 * ((self.enc_depth ** 4) * self.dec_depth) ** -0.0625)
            deepnorm_init(self.decoder, (12 * self.dec_depth) ** -0.25)

        def forward_without_retrieval(self, seq):
            # embed sequence

            embed = self.token_emb(seq)
            embed = embed[:, :self.seq_len]

            # get absolute positional embedding

            pos_emb = pos_emb(jnp.arange(embed.shape[1]))
            pos_emb = rearrange(pos_emb, 'n d -> 1 n d')
            embed = embed + pos_emb

            embed = to_decoder_model_dim(embed)
            embed = decoder(embed)

            # project to logits

            return to_logits(embed)

        """
        b - batch
        n - sequence length / chunk length
        k - number of chunks
        d - feature dimension
        r - num retrieved neighbors
        """

        if not exists(retrieved):
            return self.forward_without_retrieval(seq)

        assert not (return_loss and not self.training), 'must be training if returning loss'

        # assume padding token id (usually 0.) is to be masked out

        mask = retrieved != self.pad_id

        # handle some user inputs

        if retrieved.ndim == 3:
            retrieved = rearrange(retrieved, 'b k n -> b k 1 n') # 1 neighbor retrieved

        # if training, derive labels

        if return_loss:
            seq, labels = seq[:, :-1], seq[:, 1:]

        # variables

        n, num_chunks, num_neighbors, chunk_size, retrieved_shape, device = seq.shape[-1], *retrieved.shape[-3:], retrieved.shape, seq.device

        assert chunk_size >= self.chunk_size, 'chunk size of retrieval input must be greater or equal to the designated chunk_size on RETRO initialization'

        num_seq_chunks = n // self.chunk_size
        assert num_chunks == num_seq_chunks, f'sequence requires {num_seq_chunks} retrieved chunks, but only {num_chunks} passed in'

        # sequence index at which k-nearest neighbors have not been fetched yet after

        seq_index = num_seq_chunks * self.chunk_size

        # embed both sequence and retrieved chunks

        embed = token_emb(seq)
        retrieved = token_emb(retrieved)

        # get absolute positional embedding

        pos_emb = pos_emb(jnp.arange(n))
        pos_emb = rearrange(pos_emb, 'n d -> 1 n d')
        embed = embed + pos_emb

        # handle masks for encoder and decoder, if needed

        encoder_retrieved_mask = decoder_retrieved_mask = None

        if exists(mask):
            assert mask.shape == retrieved_shape, 'retrieval mask must be of the same shape as the retrieval tokens'
            encoder_retrieved_mask = rearrange(mask, 'b k r n -> (b k r) n')
            decoder_retrieved_mask = mask

        # project both sequence embedding and retrieved embedding to decoder dimension if necessary

        embed = to_decoder_model_dim(embed)

        # decode

        embed = decoder(
            embed,
            encoder = encoder,
            context_mask = decoder_retrieved_mask,
            encoder_retrieved_mask = encoder_retrieved_mask,
            retrieved = retrieved
        )

        # project to logits

        logits = to_logits(embed)

        if not return_loss:
            return logits

        # cross entropy loss

        loss = nn.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels, ignore_index = self.pad_id)
        return loss