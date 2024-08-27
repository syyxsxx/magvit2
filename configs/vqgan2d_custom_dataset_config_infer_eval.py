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

# pylint: disable=line-too-long
r"""Configs for the VQGAN-2D on the UCF101.

"""

import ml_collections

CUSTOM_DATASET_TRAIN_SIZE = 120000
CUSTOM_DATASET_TEST_SIZE = 30000
NUM_CLASSES = 2
VARIANT = 'VQGAN/2D'


def get_config(config_str='H'):
  """Returns the base experiment configuration."""
  version, *options = config_str.split('-')

  config = ml_collections.ConfigDict()
  config.experiment_name = f'custom_dataset_{VARIANT}'

  # Overall
  config.rng_seed = 0
  config.image_size = 128
  config.batch_size = 18
  config.eval_batch_size = config.get_ref('batch_size') // 4
  config.num_training_epochs = 100
  config.lax_precision = 'default'

  # Dataset.
  config.dataset_name = 'custom_dataset'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.num_classes = NUM_CLASSES
  config.dataset_configs.num_frames = 17
  config.dataset_configs.stride = 1
  # TODO(Lijun-Yu, roadjiang): test this augmentation and other augmentations.
  # min_resize = int(min_resize / 224 * 256)
  config.dataset_configs.min_resize = config.get_oneway_ref('image_size')
  config.dataset_configs.crop_size = config.get_ref('image_size')
  config.dataset_configs.one_hot_label = False
  config.dataset_configs.zero_centering = False  # Range is 0 to 1
  config.dataset_configs.num_test_clips = 1
  config.dataset_configs.prefetch_to_device = 2
  config.dataset_configs.dataset_path = "/workspace/custom_dataload/custom_dataset/"
  config.dataset_configs.shuffle_buffer_size = 10_000
  config.num_train_sampled_frames = 4

  # Model: vqvae
  model_class, model_type = VARIANT.split('/')
  config.dtype = config.get_ref('data_dtype_str')
  config.model_class = model_class

  config.pretrained_image_model = True
  config.perceptual_loss_weight = 0.1
  config.perceptual_loss_on_logit = True
  config.polyak_decay = 0.999  # ema decay factor for generator

  config.vqgan = ml_collections.ConfigDict()
  config.vqgan.model_type = model_type
  config.vqgan.loss_type = 'non-saturating'
  config.vqgan.g_adversarial_loss_weight = 0.1
  config.vqgan.gradient_penalty = 'r1'  #  r1, none
  config.vqgan.grad_penalty_cost = 10.0
  # finetune decoder
  config.vqgan.finetune_decoder = False
  config.vqgan.finetune_path = ''

  config.vqvae = ml_collections.ConfigDict()
  config.vqvae.architecture = '2dcnn'
  config.vqvae.codebook_size = 4096
  config.vqvae.entropy_loss_ratio = 0.1
  config.vqvae.entropy_temperature = 0.01
  config.vqvae.entropy_loss_type = 'softmax'
  config.vqvae.commitment_cost = 0.25

  config.vqvae.filters = {'B': 64, 'L': 128, 'H': 128}[version]
  config.vqvae.num_enc_res_blocks = 4
  config.vqvae.num_dec_res_blocks = 4
  config.vqvae.num_enc_remat_blocks = 3
  config.vqvae.num_dec_remat_blocks = 3
  config.vqvae.channel_multipliers = (1, 2, 2, 4)
  config.vqvae.embedding_dim = 512
  config.vqvae.conv_downsample = True
  config.vqvae.deconv_upsample = False
  config.vqvae.activation_fn = 'swish'
  config.vqvae.norm_type = 'GN'

  config.discriminator = ml_collections.ConfigDict()
  config.discriminator.filters = config.vqvae.get_oneway_ref('filters')
  config.discriminator.channel_multipliers = (2, 4, 4, 4, 4)
  config.discriminator.num_remat_blocks = 3

  # Learning rate.
  config.base_lr = 0.0001
  config.optimizer = ml_collections.ConfigDict()
  config.optimizer.lr = config.get_ref('base_lr')
  config.optimizer.beta1 = 0.0
  config.optimizer.beta2 = 0.99

  config.optimizer.g_lr = config.get_ref('base_lr')
  config.optimizer.d_lr = config.get_ref('base_lr')

  steps_per_epoch = CUSTOM_DATASET_TRAIN_SIZE // config.get_ref('batch_size')
  total_steps = config.get_ref('num_training_epochs') * steps_per_epoch
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = 5 * steps_per_epoch
  config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.base_learning_rate = config.get_ref('base_lr')  # placeholder

  # Pretrained model on ImageNet.
  config.init_from = ml_collections.ConfigDict()
  config.init_from.checkpoint_path = {
      'B': 'gs://magvit/models/imagenet_2d_base',
      'L': 'not_support_yet',
      'H': 'gs://magvit/models/imagenet_2d_base'
  }[version]

  # Evaluation.
  config.eval = ml_collections.ConfigDict()
  config.eval.enable_inception_score = False
  config.eval.enable_frechet_distance = False
  config.eval.data_splits = 'validation'
  config.eval.num_examples = 1
  config.eval.final_num_repeats = 1
  config.eval.results_dir = "./results"
  config.eval_from = ml_collections.ConfigDict()
  config.eval_from.checkpoint_path = './dircu00'
  config.eval_from.step = None

  # Logging.
  config.logging = ml_collections.ConfigDict()
  config.logging.enable_checkpoint = True
  config.logging.checkpoint_steps = 5000
  config.logging.checkpoint_kept = 5
  config.logging.log_metric_steps = 100
  config.logging.log_sample_size = 2

  if 'runlocal' in options:
    config.batch_size = 2
    config.num_training_epochs = 10
    config.eval_batch_split = 1

    config.pretrained_image_model = False
    config.perceptual_loss_weight = 0.0

    config.vqvae.filters = 32
    config.vqvae.num_enc_res_blocks = 1
    config.vqvae.num_dec_res_blocks = 1
    config.vqvae.channel_multipliers = (1,) * len(
        config.vqvae.channel_multipliers)
    config.discriminator.channel_multipliers = (1,) * len(
        config.discriminator.channel_multipliers)

    config.logging.enable_checkpoint = False
    config.logging.checkpoint_steps = 100
    config.logging.log_metric_steps = 20

    del config.init_from

  # Standalone evaluation.
  if 'eval' in options:
    config.eval_only = True
    config.eval_from.checkpoint_path = ''
    config.eval_from.step = -1
    config.eval_from.legacy_checkpoint = True

  return config

