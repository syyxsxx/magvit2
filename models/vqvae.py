# Copyright 2023 The videogvt Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VQVAE 3D Model."""
from typing import Any, Dict, Tuple, Type, Union, Sequence, Optional
from math import log2, ceil

from absl import logging
import flax.linen as nn

import jax
import jax.nn as jnn
import jax.numpy as jnp
import ml_collections
from numpy import typing as nptyping
from einops import rearrange, reduce, pack, unpack

from videogvt.models import enc_dec_2dcnn
from videogvt.models import enc_dec_2plus1dcnn
from videogvt.models import enc_dec_3dcnn
from videogvt.models import model_utils
from videogvt.train_lib import losses


ArrayLike = Union[jax.typing.ArrayLike, Sequence['ArrayLike']]
DTypeLike = nptyping.DTypeLike

def default(*args):
  for arg in args:
    if exists(arg):
      return arg() if callable(arg) else arg
  return None

def exists(v):
  return v is not None

def pack_one(t, pattern):
  return pack([t], pattern)

def mse_loss(input, target):
  return jnp.mean(jnp.square(input - target), axis=-1)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def entropy(prob, eps=1e-12):
  prob = jnp.clip(prob, eps, 1.0 - eps)  # 防止log(0)的情况
  return -(prob * jnp.log(prob)).sum(axis=-1)

def l2_normalize(x, axis=None, epsilon=1e-12):
  square_sum = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
  x_inv_norm = jax.lax.rsqrt(jnp.maximum(square_sum, epsilon))
  return jnp.multiply(x, x_inv_norm)

class LFQuantizer(nn.Module):
  """Lookup Free Quantizer"""
  config: ml_collections.ConfigDict
  dtype: int = jnp.float32
  precision: Any = jax.lax.Precision.DEFAULT
  num_codebooks: int = 1

  def setup(self):
    """LFQ setup."""
    codebook_size = self.config.vqvae.codebook_size
    dim = self.config.vqvae.embedding_dim
    keep_num_codebooks_dim = None
    self.dim = dim
    self.codebook_dims = int(log2(codebook_size))
    self.has_projections = self.dim != self.codebook_dims
    if self.has_projections:
      self.project_in = nn.Dense(features=self.codebook_dims)
      self.project_out = nn.Dense(features=self.dim)
    keep_num_codebooks_dim = default(keep_num_codebooks_dim, self.num_codebooks > 1)
    assert not (self.num_codebooks > 1 and not keep_num_codebooks_dim)
    self.keep_num_codebooks_dim = keep_num_codebooks_dim
    self.codebook_scale = 1.0
    self.mask = 2 ** jnp.arange(self.codebook_dims - 1, -1, -1)
    print("############mask###########")
    print(jax.device_get(self.mask))
    self.zero = jnp.array(0.)
    self.frac_per_sample_entropy = 1
    self.diversity_gamma = 1.0
    self.entropy_loss_weight = self.config.vqvae.entropy_loss_ratio
    self.commitment_loss_weight = self.config.vqvae.commitment_cost
    all_codes = jnp.arange(codebook_size)
    bits = ((all_codes[..., None].astype(jnp.int32) & self.mask) != 0).astype(jnp.float32)
    self.codebook = self.bits_to_codes(bits)

  def bits_to_codes(self, bits):
    return bits * self.codebook_scale * 2 - self.codebook_scale

  def decode_ids(self, indices):
    if not self.keep_num_codebooks_dim:
      indices = rearrange(indices, '... -> ... 1')
    bits = ((indices[..., None].astype(jnp.int32) & self.mask) != 0)
    codes = self.bits_to_codes(bits)
    codes = rearrange(codes, '... c d -> ... (c d)')
    if self.has_projections:
      codes = self.project_out(codes)
    #codes = rearrange(codes, 'b ... d -> b d ...')
    return codes

  def __call__(self, x, *, is_train=False):
    print("quantized-----------step0")
    print(x.shape)
    x, ps = pack_one(x, 'b * d')
    print("quantized-----------step1")
    print(x.shape)
    if self.has_projections:
      x = self.project_in(x)
    print("quantized-----------step2")
    print(x.shape)
    x = rearrange(x, 'b n (c d) -> b n c d', c = self.num_codebooks)
    original_input = x
    codebook_value = jnp.ones_like(x) * self.codebook_scale
    quantized = jnp.where(x > 0, codebook_value, -codebook_value)
    if is_train:
      x = x + jax.lax.stop_gradient(quantized - x)
    else:
      x = quantized
    indices = reduce((x > 0).astype(jnp.int32) * self.mask.astype(jnp.int32), 'b n c d -> b n c', 'sum')
    # entropy aux loss
    if is_train:
      distance = -2 * jnp.einsum('... i d, j d -> ... i j', original_input, self.codebook)
      inv_temperature = 100
      prob = jnn.softmax(-distance * inv_temperature, axis=-1)
      prob = rearrange(prob, 'b n ... -> (b n) ...')
      per_sample_probs = prob
      per_sample_entropy = entropy(per_sample_probs).mean()
      avg_prob = reduce(per_sample_probs, '... c d -> c d', 'mean')
      codebook_entropy = entropy(avg_prob).mean()
      entropy_aux_loss = per_sample_entropy - self.diversity_gamma * codebook_entropy
    else:
      # if not training, just return dummy 0
      entropy_aux_loss = per_sample_entropy = codebook_entropy = self.zero
    if is_train:
      commit_loss = mse_loss(original_input, jax.lax.stop_gradient(quantized))
      commit_loss = commit_loss.mean()
    else:
      commit_loss = self.zero
    x = rearrange(x, 'b n c d -> b n (c d)')
    if self.has_projections:
      x = self.project_out(x)
    x = unpack_one(x, ps, 'b * d')
    #x = rearrange(x, 'b ... d -> b d ...')
    indices = unpack_one(indices, ps, 'b * c')
    if not self.keep_num_codebooks_dim:
      indices = rearrange(indices, '... 1 -> ...')
    aux_loss = entropy_aux_loss * self.entropy_loss_weight + commit_loss * self.commitment_loss_weight
    result_dict = dict(
      quantizer_loss=aux_loss,
      entropy_loss=entropy_aux_loss)
    result_dict.update({
      'encoding_indices': indices,
      'raw': x,
    })
    return x, result_dict

class FLFQuantizer(nn.Module):
  """Cookbook Frozen Basic vector quantizer."""
  config: ml_collections.ConfigDict
  dtype: int = jnp.float32
  precision: Any = jax.lax.Precision.DEFAULT

  def setup(self):
    self.codebook_size = self.config.vqvae.codebook_size
    self.codebook_dims = int(log2(self.codebook_size))
    self.dim = self.config.vqvae.embedding_dim
    self.project_in = nn.Dense(features=self.codebook_dims)
    self.project_out = nn.Dense(features=self.dim)
    mask = 2 ** jnp.arange(self.codebook_dims - 1, -1, -1)
    all_codes = jnp.arange(self.codebook_size)
    bits = ((all_codes[..., None].astype(jnp.int32) & mask) != 0).astype(jnp.float32)
    self.codebook = bits * 1.0 * 2 - 1.0

  @nn.compact
  def __call__(self, x, *, is_train=False):
    codebook_size = self.codebook_size
    x = self.project_in(x)
    codebook = self.codebook
    if self.config.vqvae.get('latent_normalize', False):
      x = l2_normalize(x, axis=-1)
      codebook = l2_normalize(codebook, axis=-1)
    distances = jnp.reshape(
        losses.squared_euclidean_distance(
            jnp.reshape(x, (-1, x.shape[-1])), codebook),
        x.shape[:-1] + (codebook_size,))
    encoding_indices = jnp.argmin(distances, axis=-1)
    encodings = jax.nn.one_hot(
        encoding_indices, codebook_size, dtype=self.dtype)
    quantized = self.quantize(encodings)
    result_dict = dict()
    if is_train:
      e_latent_loss = jnp.mean((jax.lax.stop_gradient(quantized) - x)**
                               2) * self.config.vqvae.commitment_cost
      q_latent_loss = jnp.mean(
          (quantized - jax.lax.stop_gradient(x))**2) * self.config.vqvae.get(
              'embedding_loss_ratio', 1.)
      entropy_loss = 0.0
      if self.config.vqvae.entropy_loss_ratio != 0:
        entropy_loss = losses.entropy_loss(
            -distances,
            loss_type=self.config.vqvae.entropy_loss_type,
            temperature=self.config.vqvae.entropy_temperature
        ) * self.config.vqvae.entropy_loss_ratio
      e_latent_loss = jnp.asarray(e_latent_loss, dtype=self.dtype)
      q_latent_loss = jnp.asarray(q_latent_loss, dtype=self.dtype)
      entropy_loss = jnp.asarray(entropy_loss, dtype=self.dtype)
      loss = e_latent_loss + q_latent_loss + entropy_loss
      result_dict = dict(
          quantizer_loss=loss,
          e_latent_loss=e_latent_loss,
          q_latent_loss=q_latent_loss,
          entropy_loss=entropy_loss)
      quantized = x + jax.lax.stop_gradient(quantized - x)
    quantized = self.project_out(quantized)
    result_dict.update({
        'encodings': encodings,
        'encoding_indices': encoding_indices,
        'raw': x,
    })
    return quantized, result_dict

  def quantize(self, z: jnp.ndarray) -> jnp.ndarray:
    codebook = self.codebook
    return jnp.dot(z, codebook, precision=self.precision)

  def decode_ids(self, ids: jnp.ndarray) -> jnp.ndarray:
    codebook = self.codebook
    return jnp.take(codebook, ids, axis=0)



class VectorQuantizer(nn.Module):
  """Basic vector quantizer."""
  config: ml_collections.ConfigDict
  dtype: int = jnp.float32
  precision: Any = jax.lax.Precision.DEFAULT

  @nn.compact
  def __call__(self, x, *, is_train=False):
    codebook_size = self.config.vqvae.codebook_size
    codebook = self.param(
        'codebook',
        jax.nn.initializers.variance_scaling(
            scale=1.0, mode='fan_in', distribution='uniform'),
        (codebook_size, x.shape[-1]))
    codebook = jnp.asarray(codebook, dtype=self.dtype)
    if self.config.vqvae.get('latent_normalize', False):
      x = l2_normalize(x, axis=-1)
      codebook = l2_normalize(codebook, axis=-1)
    distances = jnp.reshape(
        losses.squared_euclidean_distance(
            jnp.reshape(x, (-1, x.shape[-1])), codebook),
        x.shape[:-1] + (codebook_size,))
    encoding_indices = jnp.argmin(distances, axis=-1)
    encodings = jax.nn.one_hot(
        encoding_indices, codebook_size, dtype=self.dtype)
    quantized = self.quantize(encodings)
    result_dict = dict()
    if is_train:
      e_latent_loss = jnp.mean((jax.lax.stop_gradient(quantized) - x)**
                               2) * self.config.vqvae.commitment_cost
      q_latent_loss = jnp.mean(
          (quantized - jax.lax.stop_gradient(x))**2) * self.config.vqvae.get(
              'embedding_loss_ratio', 1.)
      entropy_loss = 0.0
      if self.config.vqvae.entropy_loss_ratio != 0:
        entropy_loss = losses.entropy_loss(
            -distances,
            loss_type=self.config.vqvae.entropy_loss_type,
            temperature=self.config.vqvae.entropy_temperature
        ) * self.config.vqvae.entropy_loss_ratio
      e_latent_loss = jnp.asarray(e_latent_loss, dtype=self.dtype)
      q_latent_loss = jnp.asarray(q_latent_loss, dtype=self.dtype)
      entropy_loss = jnp.asarray(entropy_loss, dtype=self.dtype)
      loss = e_latent_loss + q_latent_loss + entropy_loss
      result_dict = dict(
          quantizer_loss=loss,
          e_latent_loss=e_latent_loss,
          q_latent_loss=q_latent_loss,
          entropy_loss=entropy_loss)
      quantized = x + jax.lax.stop_gradient(quantized - x)
    result_dict.update({
        'encodings': encodings,
        'encoding_indices': encoding_indices,
        'raw': x,
    })
    return quantized, result_dict

  def quantize(self, z: jnp.ndarray) -> jnp.ndarray:
    codebook = self.get_codebook()
    return jnp.dot(z, codebook, precision=self.precision)

  def get_codebook(self) -> jnp.ndarray:
    return jnp.asarray(self.variables['params']['codebook'], dtype=self.dtype)

  def decode_ids(self, ids: jnp.ndarray) -> jnp.ndarray:
    codebook = self.get_codebook()
    return jnp.take(codebook, ids, axis=0)




class VQVAE(nn.Module):
  """VQ-VAE model."""
  config: ml_collections.ConfigDict
  dtype: int = jnp.float32
  activation_fn: Any = nn.relu
  precision: Any = jax.lax.Precision.DEFAULT

  def setup(self):
    """VQ-VAE setup."""
    quantizer_str = self.config.vqvae.get(
        'vector_quantizer_class', 'VectorQuantizer')
    if quantizer_str == 'VectorQuantizer':
      self.quantizer = VectorQuantizer(
          config=self.config, precision=self.precision, dtype=self.dtype
      )
    elif quantizer_str == 'LFQuantizer':
      self.quantizer = LFQuantizer(
          config=self.config, precision=self.precision, dtype=self.dtype, num_codebooks=1
      )
    elif quantizer_str == 'FLFQuantizer':
      self.quantizer = FLFQuantizer(
          config=self.config, precision=self.precision, dtype=self.dtype
      )
    else:
      raise NotImplementedError(quantizer_str)

    if self.config.vqvae.architecture == '2dcnn':
      self.encoder = model_utils.vmap_t_dim(enc_dec_2dcnn.Encoder)(
          config=self.config, dtype=self.dtype)
      self.decoder = model_utils.vmap_t_dim(enc_dec_2dcnn.Decoder)(
          config=self.config, output_dim=3)
    elif self.config.vqvae.architecture == '3dcnn':
      self.encoder = enc_dec_3dcnn.Encoder(config=self.config, dtype=self.dtype)
      self.decoder = enc_dec_3dcnn.Decoder(config=self.config, output_dim=3)
    elif self.config.vqvae.architecture == '2plus1dcnn':
      self.encoder = enc_dec_2plus1dcnn.Encoder(
          config=self.config, dtype=self.dtype)
      self.decoder = enc_dec_2plus1dcnn.Decoder(
          config=self.config, output_dim=3)
    else:
      raise NotImplementedError(
          f'Architecture {self.config.vqvae.architecture}')

  def encode(
      self,
      x: jnp.ndarray,
      *,
      is_train: bool = False) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    video = x
    encoded_feature = self.encoder(video, is_train=is_train)
    print("#############encoded_feature dim################")
    print(encoded_feature.shape)
    quantized, result_dict = self.quantizer(encoded_feature, is_train=is_train)
    return quantized, result_dict  # pytype: disable=bad-return-type  # jax-ndarray

  def decode(self, x: jnp.ndarray) -> jnp.ndarray:
    return self.decoder(x, is_train=False)

  def get_codebook_funct(self):
    # This function only works for the naive VQGAN
    return self.quantizer.get_codebook()

  def decode_from_indices(self, ids: jnp.ndarray) -> jnp.ndarray:
    features = self.quantizer.decode_ids(ids)
    reconstructed_video = self.decode(features)
    return reconstructed_video

  def decode_stage1(self, ids: jnp.ndarray) -> jnp.ndarray:
    assert self.config.vqvae.architecture == '3dcnn', 'Only support 3dcnn.'
    features = self.quantizer.decode_ids(ids)
    pre_activation_embeddings = self.decoder(
        features, is_train=False, mode='stage1')
    return pre_activation_embeddings

  def decode_stage2(self, embeddings: jnp.ndarray) -> jnp.ndarray:
    assert self.config.vqvae.architecture == '3dcnn', 'Only support 3dcnn.'
    reconstructed_video = self.decoder(
        embeddings, is_train=False, mode='stage2')
    return reconstructed_video

  def encode_to_indices(self, inputs: jnp.ndarray) -> jnp.ndarray:
    _, result_dict = self.encode(inputs, is_train=False)
    ids = result_dict['encoding_indices']
    return ids

  def __call__(
      self,
      input_video: jnp.ndarray,
      *,
      is_train: bool = False) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    quantized, result_dict = self.encode(input_video, is_train=is_train)
    print("xxxxxxxxxxxxquantizedxxxxxxxxxxxxxx")
    print(quantized.shape)
    outputs = self.decoder(quantized, is_train=is_train)
    return outputs, result_dict

