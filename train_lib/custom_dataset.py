

import functools
import tensorflow as tf
import jax.numpy as jnp
from typing import Optional
from scenic.dataset_lib import dataset_utils as scenic_dataset_utils
from scenic.dataset_lib import datasets as scenic_datasets
from flax import jax_utils
from dmvr import processors


# from videogvt.train_lib import dataset_utils


def preprocess_train_example(example,
                             dtype=tf.float32,
                             num_frames=16,
                             stride=1,
                             zero_centering=False,
                             image_size=128):
    """Preprocesses the given video.

  Args:
    example: dict; Example that has an 'image_main'.
    camera_name: Name of the image sequence to use.
    dtype: Tensorflow data type; Data type of the image.
    zero_centering: If True, frames are normalized to values in [-1, 1].
      If False, values in [0, 1].

  Returns:
    dict; Example that has an 'inputs'.
  """
    frames = example['video']
    frames = processors.sample_linspace_sequence(frames, 1, num_frames,
                                                 stride)
    frames = processors.resize_smallest(frames, image_size)
    frames = processors.crop_image(frames, image_size, image_size, random=True)
    frames = processors.normalize_image(frames, zero_centering, dtype)
    return {'inputs': frames}


def preprocess_eval_example(example,
                            dtype=tf.float32,
                            num_frames=16,
                            stride=1,
                            num_clips=1,
                            zero_centering=True,
                            image_size=128):
    frames = example['video']
    frames = processors.sample_linspace_sequence(frames, num_clips, num_frames,
                                                 stride)
    frames = processors.resize_smallest(frames, image_size)
    frames = processors.crop_image(frames, image_size, image_size)
    frames = processors.normalize_image(frames, zero_centering, dtype)
    return {'inputs': frames}


@scenic_datasets.add_dataset('custom_dataset')
def get_dataset(*,
                batch_size,
                eval_batch_size,
                num_shards,
                dtype_str='float32',
                shuffle_seed=0,
                rng=None,
                dataset_configs=None,
                dataset_service_address: Optional[str] = None):
    """Returns generators for the BAIR train, validation, and test set.

  Args:
    batch_size: int; Determines the train batch size.
    eval_batch_size: int; Determines the evaluation batch size.
    num_shards: int;  Number of shards --> batch shape: [num_shards, bs, ...].
    dtype_str: Data type of the image (e.g. 'float32').
    shuffle_seed: int; Seed for shuffling the training data.
    rng: JAX rng key, which can be used for augmentation, shuffling, etc.
    dataset_configs: dict; Dataset specific configurations.
    dataset_service_address: If set, will distribute the training dataset using
      the given tf.data service at the given address.

  Returns:
    A dataset_utils.Dataset() which includes a train_iter, a valid_iter,
      a test_iter, and a dict of meta_data.
  """
    del rng
    dtype = getattr(tf, dtype_str)
    dataset_configs = dataset_configs or {}
    num_frames = dataset_configs.get('num_frames', 16)
    # need to fps / 8
    stride = dataset_configs.get('stride', 1)
    zero_centering = dataset_configs.get('zero_centering', True)
    num_eval_clips = dataset_configs.get('num_eval_clips', 1)
    shuffle_buffer_size = dataset_configs.get('shuffle_buffer_size', None)
    dataset_path = dataset_configs.get('dataset_path', "")
    image_size = 128

    preprocess_train = functools.partial(
        preprocess_train_example,
        dtype=dtype,
        num_frames=num_frames,
        stride=stride,
        zero_centering=zero_centering,
        image_size=image_size)
    preprocess_eval = functools.partial(
        preprocess_eval_example,
        dtype=dtype,
        num_frames=num_frames,
        stride=stride,
        num_clips=num_eval_clips,
        zero_centering=zero_centering,
        image_size=image_size)
    # 这里要确定split的情况
    print("***"*10)
    print(dataset_path)
    train_ds, _ = scenic_dataset_utils.load_split_from_tfds(
        'custom_dataset',
        batch_size,
        split='train',
        preprocess_example=preprocess_train,
        augment_train_example=None,
        shuffle_buffer_size=shuffle_buffer_size,
        shuffle_seed=shuffle_seed,
        dataset_path=dataset_path,
        cache=False)
    eval_ds, _ = scenic_dataset_utils.load_split_from_tfds(
        'custom_dataset',
        eval_batch_size,
        split='test',
        preprocess_example=preprocess_eval,
        dataset_path=dataset_path,
        cache=False)

    maybe_pad_batches_train = functools.partial(
        scenic_dataset_utils.maybe_pad_batch, train=True, batch_size=batch_size)
    maybe_pad_batches_eval = functools.partial(
        scenic_dataset_utils.maybe_pad_batch,
        train=False,
        batch_size=eval_batch_size * num_eval_clips)
    shard_batches = functools.partial(scenic_dataset_utils.shard, n_devices=num_shards)

    train_iter = iter(train_ds)
    train_iter = map(scenic_dataset_utils.tf_to_numpy, train_iter)
    train_iter = map(maybe_pad_batches_train, train_iter)
    train_iter = map(shard_batches, train_iter)
    if dataset_configs.get('prefetch_to_device'):
        # Async bind batch to device which speeds up training.
        train_iter = jax_utils.prefetch_to_device(
            train_iter, dataset_configs.get('prefetch_to_device'))

    eval_iter = iter(eval_ds)
    eval_iter = map(scenic_dataset_utils.tf_to_numpy, eval_iter)
    eval_iter = map(maybe_pad_batches_eval, eval_iter)
    eval_iter = map(shard_batches, eval_iter)

    input_shape = (-1, num_frames, 128, 128, 3)
    num_train_examples = scenic_dataset_utils.get_num_examples(
        'custom_dataset', 'train')
    num_eval_examples = scenic_dataset_utils.get_num_examples('custom_dataset',
                                                              'test') * num_eval_clips
    meta_data = {
        'num_classes': None,
        'input_shape': input_shape,
        'num_train_examples': num_train_examples,
        'num_eval_examples': num_eval_examples,
        'input_dtype': getattr(jnp, dtype_str),
        'target_is_onehot': False,
    }
    print("***"*10)
    print("meta_data:", meta_data)
    return scenic_dataset_utils.Dataset(train_iter, eval_iter, None, meta_data)

