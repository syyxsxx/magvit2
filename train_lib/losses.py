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

"""Common losses used in GANs."""
from typing import Any, Optional

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from videogvt.train_lib import pretrained_model_utils

DEFAULT_PRECISION = jax.lax.Precision.HIGHEST


def squared_euclidean_distance(a: jnp.ndarray,  # pytype: disable=annotation-type-mismatch  # jax-ndarray
                               b: jnp.ndarray,
                               b2: jnp.ndarray = None,
                               precision=DEFAULT_PRECISION) -> jnp.ndarray:
  """Computes the pairwise squared Euclidean distance.

  Args:
    a: float32: (n, d): An array of points.
    b: float32: (m, d): An array of points.
    b2: float32: (d, m): b square transpose.
    precision: use HIGHEST precision by default

  Returns:
    d: float32: (n, m): Where d[i, j] is the squared Euclidean distance between
    a[i] and b[j].
  """
  if b2 is None:
    b2 = jnp.sum(b.T**2, axis=0, keepdims=True)
  a2 = jnp.sum(a**2, axis=1, keepdims=True)
  ab = jnp.matmul(a, b.T, precision=precision)
  d = a2 - 2 * ab + b2
  return d


def sequence_soft_cross_entropy_loss(*, target_labels: jnp.ndarray,
                                     target_probs: jnp.ndarray,
                                     logits: jnp.ndarray) -> jnp.ndarray:
  """Computes the mean cross-entropy for the sequence soft-label predictions.

  Args:
    target_labels: 3D int array of shape (B, T, K) where each value is in [0,
      C-1].
    target_probs: 3D array of shape (B, T, K) where K is number of top
      centroids.
    logits: 3D array of shape (B, T, C) where C is number of classes.

  Returns:
    float loss.
  """
  logprobs = jax.nn.log_softmax(logits, axis=-1)
  cross_entropy = jnp.sum(
      target_probs * jnp.take_along_axis(logprobs, target_labels, axis=-1),
      axis=-1)
  return -jnp.mean(cross_entropy)


def sequence_cross_entropy_loss(*, labels: jnp.ndarray,
                                logits: jnp.ndarray) -> jnp.ndarray:
  """Computes the mean cross-entropy for the sequence predictions.

  Args:
    labels: 2D int array of shape (B, T) where each value is in [0, C-1].
    logits: 3D array of shape (B, T, C) where C is number of classes.

  Returns:
    float loss.
  """
  _, _, num_classes = logits.shape
  labels_one_hot = jax.nn.one_hot(labels, num_classes)
  logprobs = jax.nn.log_softmax(logits, axis=-1)
  return -jnp.mean(jnp.sum(logprobs * labels_one_hot, axis=-1))


@jax.vmap
def sigmoid_cross_entropy_with_logits(*, labels: jnp.ndarray,
                                      logits: jnp.ndarray) -> jnp.ndarray:
  """Sigmoid cross entropy loss.

  We use a stable formulation that is equivalent to the one used in TensorFlow.
  The following derivation shows how we arrive at the formulation:

  .. math::
        z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
      = (1 - z) * x + log(1 + exp(-x))
      = x - x * z + log(1 + exp(-x))

  For x < 0, the following formula is more stable:
  .. math::
        x - x * z + log(1 + exp(-x))
      = log(exp(x)) - x * z + log(1 + exp(-x))
      = - x * z + log(1 + exp(x))

  We combine the two cases (x<0, x>=0) into one formula as follows:
  .. math::
      max(x, 0) - x * z + log(1 + exp(-abs(x)))

  This function is vmapped, so it is written for a single example, but can
  handle a batch of examples.

  Args:
    labels: The correct labels.
    logits: The output logits.

  Returns:
    The binary cross entropy loss for each given logit.
  """
  # The final formulation is: max(x, 0) - x * z + log(1 + exp(-abs(x)))
  # To allow computing gradients at zero, we define custom versions of max and
  # abs functions just like in tf.nn.sigmoid_cross_entropy_with_logits.
  zeros = jnp.zeros_like(logits, dtype=logits.dtype)
  condition = (logits >= zeros)
  relu_logits = jnp.where(condition, logits, zeros)
  neg_abs_logits = jnp.where(condition, -logits, logits)
  return relu_logits - logits * labels + jnp.log1p(jnp.exp(neg_abs_logits))


def apply_label_smoothing(one_hot_targets: jnp.ndarray,
                          label_smoothing: float) -> jnp.ndarray:
  """Apply label smoothing to the one-hot targets.

  Applies label smoothing such that the on-values are transformed from 1.0 to
  `1.0 - label_smoothing + label_smoothing / num_classes`, and the off-values
  are transformed from 0.0 to `label_smoothing / num_classes`.
  https://arxiv.org/abs/1512.00567

  Note that another way of performing label smoothing (which we don't use here)
  is to take `label_smoothing` mass from the on-values and distribute it to the
  off-values; in other words, transform the on-values to `1.0 - label_smoothing`
  and the  off-values to `label_smoothing / (num_classes - 1)`.
  http://jmlr.org/papers/v20/18-789.html


  Args:
    one_hot_targets: One-hot targets for an example, a [batch, ..., num_classes]
      float array.
    label_smoothing: A scalar in [0, 1] used to smooth the labels.

  Returns:
    A float array of the same shape as `one_hot_targets` with smoothed label
    values.
  """
  on_value = 1.0 - label_smoothing
  num_classes = one_hot_targets.shape[-1]
  off_value = label_smoothing / num_classes
  one_hot_targets = one_hot_targets * on_value + off_value
  return one_hot_targets


def weighted_sequence_cross_entropy_loss(
    *,
    labels: jnp.ndarray,
    logits: jnp.ndarray,
    weights: jnp.ndarray,
    label_smoothing: Optional[float] = 0.0):
  """Computes the mean cross-entropy for the sequence predictions.

  Args:
    labels: 2D int array of shape (B, T) where each value is in [0, C-1].
    logits: 3D array of shape (B, T, C) where C is number of classes.
    weights: 2D float array (B, T).
    label_smoothing: float.

  Returns:
    float loss.
  """
  vocab_size = logits.shape[-1]
  one_hot_targets = jax.nn.one_hot(labels, vocab_size)
  soft_targets = apply_label_smoothing(one_hot_targets, label_smoothing)
  loss = -jnp.sum(soft_targets * jax.nn.log_softmax(logits), axis=-1)
  loss = jnp.sum(
      loss * weights, axis=-1) / jnp.clip(jnp.sum(weights, axis=-1), 1e-8)
  return jnp.mean(loss)


def l1_loss(y_true, y_pred):
  diff = y_true - y_pred
  diff = jnp.asarray(diff, jnp.float32)
  return jnp.mean(jnp.abs(diff))


def l2_loss(y_true, y_pred, mask=None):
  diff = y_true - y_pred
  diff = jnp.asarray(diff, jnp.float32)
  if mask is not None:
    diff = diff * mask
  return jnp.mean(jnp.square(diff))


def r1_gradient_penalty(net, variables, x, penalty_cost=1.0):
  out, gradient = gradient_penalty_without_bn(net, variables, x)
  gradient = gradient.reshape((x.shape[0], -1))
  gradient = jnp.asarray(gradient, jnp.float32)
  penalty = jnp.sum(jnp.square(gradient), axis=-1)
  penalty = jnp.mean(penalty) * penalty_cost
  return out, penalty


def r1_gradient_penalty_multiinput(net, variables, x, penalty_cost=1.0):
  out, gradients = gradient_penalty_without_bn(net, variables, x)

  gradients = jax.tree_map(lambda t: t.reshape((t.shape[0], -1)), gradients)
  gradient = jnp.concatenate(jax.tree_flatten(gradients)[0], axis=-1)

  gradient = jnp.asarray(gradient, jnp.float32)
  penalty = jnp.sum(jnp.square(gradient), axis=-1)
  penalty = jnp.mean(penalty) * penalty_cost
  return out, penalty


def gradient_penalty_without_bn(net, variables, x):
  """Gradient penalty when each sample is independent.

  if in the training mode
    forward_fn = lambda x: net(train=True).apply(variables, x, mutable=True)
    out, vjp_fn, new_variables = jax.vjp(forward_fn, x, has_aux=True)
    gradient = vjp_fn(jnp.ones_like(out))[0]
    return out, gradient, new_variables
  but as we want to real and fake to have same discriminator parameters, so we
  do not want to update its parameters, so we use train mode.

  !!! discrimintor should not have BN or dropout in this case.
  Args:
    net: The network.
    variables: network variables.
    x: input image.

  Returns:
    out: output logit.
    gradient: The gradient.
  """
  forward_fn = lambda x: net(train=False).apply(variables, x, mutable=False)
  out, vjp_fn = jax.vjp(forward_fn, x, has_aux=False)
  gradient = vjp_fn(jnp.ones_like(out))[0]
  return out, gradient


def gradient_penalty(net, variables, x, train=True):  # pylint: disable=unused-argument
  """General gradient penalty."""
  del net, variables, x, train
  pass


def get_perplexity(inputs, axis_name="batch"):
  """Get perplexity."""
  feat_dim = inputs.shape[-1]
  inputs = inputs.reshape((-1, feat_dim))
  single_replica_probs = jnp.mean(inputs, axis=0)
  single_replica_probs = single_replica_probs / jnp.sum(single_replica_probs)
  single_replica_probs = jnp.asarray(single_replica_probs, jnp.float32)
  if axis_name:
    probs = jax.lax.pmean(single_replica_probs, axis_name=axis_name)
    probs = probs / jnp.sum(probs)
  else:
    probs = single_replica_probs
  perplexity = jnp.exp(-jnp.sum(probs * jnp.log(probs + 1e-5)))
  single_replica_perplexity = jnp.exp(
      -jnp.sum(single_replica_probs * jnp.log(single_replica_probs + 1e-5)))
  return perplexity, single_replica_perplexity


def entropy_loss(
    affinity,
    loss_type="softmax",
    temperature=1.0,
    sample_minimization_weight=1.0,
    batch_maximization_weight=1.0,
):
  """Calculate the entropy loss."""
  flat_affinity = affinity.reshape(-1, affinity.shape[-1])
  flat_affinity /= temperature
  probs = jax.nn.softmax(flat_affinity, axis=-1)
  log_probs = jax.nn.log_softmax(flat_affinity + 1e-5, axis=-1)
  if loss_type == "softmax":
    target_probs = probs
  elif loss_type == "argmax":
    codes = jnp.argmax(flat_affinity, axis=-1)
    onehots = jax.nn.one_hot(
        codes, flat_affinity.shape[-1], dtype=flat_affinity.dtype)
    onehots = probs - jax.lax.stop_gradient(probs - onehots)
    target_probs = onehots
  else:
    raise ValueError("Entropy loss {} not supported".format(loss_type))
  avg_probs = jnp.mean(target_probs, axis=0)
  avg_entropy = -jnp.sum(avg_probs * jnp.log(avg_probs + 1e-5))
  sample_entropy = -jnp.mean(jnp.sum(target_probs * log_probs, axis=-1))
  loss = (sample_minimization_weight * sample_entropy) - (
      batch_maximization_weight * avg_entropy
  )
  return loss


def separated_entropy(affinity, loss_type="softmax", temperature=1.0):
  """Calculate the entropy loss."""
  flat_affinity = affinity.reshape(-1, affinity.shape[-1])
  flat_affinity /= temperature
  probs = jax.nn.softmax(flat_affinity, axis=-1)
  log_probs = jax.nn.log_softmax(flat_affinity + 1e-5, axis=-1)
  if loss_type == "softmax":
    target_probs = probs
  elif loss_type == "argmax":
    codes = jnp.argmax(flat_affinity, axis=-1)
    onehots = jax.nn.one_hot(
        codes, flat_affinity.shape[-1], dtype=flat_affinity.dtype)
    onehots = probs - jax.lax.stop_gradient(probs - onehots)
    target_probs = onehots
  else:
    raise ValueError("Entropy loss {} not supported".format(loss_type))
  avg_probs = jnp.mean(target_probs, axis=0)
  avg_entropy = -jnp.sum(avg_probs * jnp.log(avg_probs + 1e-5))
  sample_entropy = -jnp.mean(jnp.sum(target_probs * log_probs, axis=-1))
  return sample_entropy, avg_entropy


def discriminator_loss(*, real_logit, fake_logit, loss_type="hinge"):
  """Adds discriminator loss."""
  if loss_type == "hinge":
    real_loss = jax.nn.relu(1.0 - real_logit)
    fake_loss = jax.nn.relu(1.0 + fake_logit)
  elif loss_type == "non-saturating":
    if real_logit is not None:
      real_loss = sigmoid_cross_entropy_with_logits(
          labels=jnp.ones_like(real_logit), logits=real_logit)
    else:
      real_loss = 0.0

    if fake_logit is not None:
      fake_loss = sigmoid_cross_entropy_with_logits(
          labels=jnp.zeros_like(fake_logit), logits=fake_logit)
    else:
      fake_loss = 0.0
  else:
    raise ValueError("Discriminator loss {} not supported".format(loss_type))
  return jnp.mean(real_loss) + jnp.mean(fake_loss)


def generator_loss(*, fake_logit, loss_type="hinge"):
  """Adds generator loss."""
  if loss_type == "hinge":
    loss = -jnp.mean(fake_logit)
  elif loss_type == "non-saturating":
    loss = jnp.mean(
        sigmoid_cross_entropy_with_logits(
            labels=jnp.ones_like(fake_logit), logits=fake_logit))
  else:
    raise ValueError("Generator loss {} not supported".format(loss_type))
  return loss


def calculate_perceptual_loss_on_pretrained(
    model: nn.Module,
    state: Any,
    real_images: jnp.ndarray,
    fake_images: jnp.ndarray,
    perceptual_loss_on_logit: bool = False):
  """Calculates perceptual loss on pre-trained model."""
  #real_pools, real_outputs = pretrained_model_utils.get_pretrained_embs(
      #state, model, images=real_images)
  #fake_pools, fake_outputs = pretrained_model_utils.get_pretrained_embs(
      #state, model, images=fake_images)
  # for https://github.com/pytorch/examples/blob/main/imagenet/main.py#L236
  real_images = jnp.asarray(real_images)
  fake_images = jnp.asarray(fake_images)
  mean = jnp.array([0.485, 0.456, 0.406])
  std = jnp.array([0.229, 0.224, 0.225])
  mean = jnp.broadcast_to(mean, real_images.shape)
  std = jnp.broadcast_to(std, real_images.shape)
  real_images = jnp.subtract(real_images, mean)
  real_images = jnp.divide(real_images, std)
  fake_images = jnp.subtract(fake_images, mean)
  fake_images = jnp.divide(fake_images, std)
  
  real_outputs = pretrained_model_utils.get_pretrained_embs(
      state, model, images=real_images)
  fake_outputs = pretrained_model_utils.get_pretrained_embs(
      state, model, images=fake_images)
  if perceptual_loss_on_logit:
    loss = l2_loss(real_outputs, fake_outputs)
  else:
    pass
    #loss = l2_loss(real_pools, fake_pools)
  return loss


def log_laplace_postprocess(inputs, laplace_eps=0.1):
  """Inverse operation of log_laplace_preprocess.

  Args:
    inputs: images of range [0, 1).
    laplace_eps: epsilon as used in log-laplace distribution.

  Returns:
    Postprocessed images for log-laplace modeling.
  """
  img = (inputs - laplace_eps) / (1.0 - 2.0 * laplace_eps)
  # Cap images in value ranges of [0, 1].
  return jnp.clip(img, 0.0, 1.0)


def log_laplace_preprocess(inputs, laplace_eps=0.1):
  """Preprocesses input images for log-laplace loss.

  Args:
    inputs: images of range [0, 1).
    laplace_eps: epsilon as used in log-laplace distribution.

  Returns:
    Preprocessed images for log-laplace modeling.
  """
  img = jnp.clip(inputs, 0.0, 1.0)
  # Convert images to [laplace_eps, 1.0 - laplace_eps).
  return img * (1.0 - 2 * laplace_eps) + laplace_eps


def log_laplace_loss(x, mu, log_sigma):
  """Computes the log laplace loss as according to ...

  equation (2) in the paper https://arxiv.org/pdf/2102.12092.pdf


  Standard laplace distribution:
  p(x1) = exp(-|x1 - mu| / sigma) / (2 * sigma), -inf < x < +inf.

  Let x1 = logit(x) = log(x / (1 - x))
  p(x) = exp(-|logit(x) - mu| / sigma) / (2 * sigma * x * (1 - x))

  loss = -log(p(x))
       = |logit(x) - mu| / sigma + log(2 * sigma * x * (1 - x))
       = |logit(x) - mu| / sigma + log(sigma) + log(2 * x * (1 - x))

  Args:
    x: tensor of shape [b, ...]. x is expected to be in range [0, 1). Typically,
      x is the groundtruth of a sampled img.
    mu: mean of the log-laplace distribution.
    log_sigma: log of sigma value above. mu and log_sigma are typically
      predictions from a model.

  Returns:
    (loss, neg_logl): a pair of tensors of shape [b, ...]. neg_logl is the
    negative log likelihood of x given p(x|mu, sigma).
  """
  # Cap x to be within [epsilon, 1.0 - epsilon].
  epsilon = 1e-6
  x = jnp.clip(x, epsilon, 1.0 - epsilon)

  logit_x = jnp.log(x / (1.0 - x))
  sigma = jnp.exp(log_sigma)
  loss = jnp.abs(logit_x - mu) / sigma + log_sigma
  # negative log likelihood needs to add in the remaining constant.
  neg_logl = loss + jnp.log(2.0 * x * (1.0 - x))
  return loss, neg_logl


def lecam_reg(real_pred, fake_pred, ema_real_pred, ema_fake_pred):
  """Lecam loss for data-efficient and stable GAN training.

  Described in https://arxiv.org/abs/2104.03310

  Args:
    real_pred: Prediction (scalar) for the real samples.
    fake_pred: Prediction for the fake samples.
    ema_real_pred: EMA prediction (scalar)  for the real samples.
    ema_fake_pred: EMA prediction for the fake samples.

  Returns:
    Lecam regularization loss (scalar).
  """
  assert real_pred.ndim == 0 and ema_fake_pred.ndim == 0
  lecam_loss = jnp.mean(jnp.power(jax.nn.relu(real_pred - ema_fake_pred), 2))
  lecam_loss += jnp.mean(jnp.power(jax.nn.relu(ema_real_pred - fake_pred), 2))
  return lecam_loss
