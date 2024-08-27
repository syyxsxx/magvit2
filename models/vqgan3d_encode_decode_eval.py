import concurrent
import functools
import itertools
import logging

import cv2
import ml_collections
import jax.numpy as jnp
import flax.linen as nn

import jax
import os
import sys
import mediapy
from flax import jax_utils

import numpy as np
from clu import metric_writers
from tensorflow.io import gfile
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scenic.dataset_lib import dataset_utils
from videogvt.configs import vqgan3d_custom_dataset_config_infer_eval
from videogvt.train_lib.eval_utils import get_train_config
from videogvt.train_lib import task_manager, inception_score, frechet_distance, image_quality_metrics, metrics_lib
from videogvt.train_lib import train_utils
from videogvt.trainers.vqgan_trainer import create_train_state
from videogvt.train_lib import train_state_manager

Batch = train_utils.Batch
TrainState = train_state_manager.VQGANTrainState
TrainStateDeprecated = train_state_manager.VQGANTrainStateDeprecated
TaskManager = task_manager.CustomTaskManager
EvalFeatureDict = metrics_lib.EvalFeatureDict

device = jax.devices()[0]
checkpoints_dir = '/workspace/dir07'
DEFAULT_HEIGHT = 128
DEFAULT_WIDTH = 128
DEFAULT_CHANNELS = 3
DEFAULT_LENGTH = 17


def get_local_batch(batch: Batch) -> Batch:
    """Slice local batch from an all_gathered batch."""
    global_bs = jax.tree_util.tree_leaves(batch)[0].shape[0]
    local_bs = global_bs // jax.process_count()
    proc_i = jax.process_index()
    return jax.tree_util.tree_map(
        lambda x: x[local_bs * proc_i:local_bs * (proc_i + 1)], batch)


def get_eval_jobs(workdir: str,
                  config: ml_collections.ConfigDict,
                  final_num_repeats: int = 1,
                  *,
                  custom_outputs: bool = False):
    """Get evaluation jobs from checkpoint manager.

    Two possible cases:
    1. When jobs run train and eval in parallel, we knew the ckpt_dir=workdir,
      and config.
    2. For a single eval job, the user needs to specify config.eval_from and
      it will look up the information for the train_config.

    Args:
      workdir: Experiment directory for saving the checkpoint.
      config: Configurations of the experiments used in the trainer.
      final_num_repeats: Number of repeats for the last checkpoint.
      custom_outputs: Whether to load custom outputs.

    Returns:
      The train_config with updated values for the specified prefixes,
        ckpt_manager, and ckpt_list.
    """
    if 'eval_from' in config:
        # separate eval job, gets info from the train experiment
        override_prefix = ('eval', 'data', 'log', 'batch',
                           'sampling') + config.eval_from.get(
            'override_prefix', tuple())
        train_config, ckpt_path = get_train_config(config, override_prefix)
        if not gfile.isdir(ckpt_path):
            ckpt_dir = os.path.dirname(ckpt_path)
        else:
            ckpt_dir = ckpt_path
    else:
        # during training/eval parallel jobs, use the current config.
        ckpt_dir = workdir
        train_config = config

    config = train_config

    if custom_outputs:
        result_dir = os.path.join(config.eval_from.result_dir,
                                  config.eval_from.get('sub_dir', ''))
        ckpt_manager = TaskManager(
            result_dir,
            workdir,
            config.eval_from.output_pattern,
            lambda x: int(  # pylint: disable=g-long-lambda
                x.split('_')[config.eval_from.get('output_step_field', -1)]
            ),
        )
    else:
        ckpt_manager = TaskManager(ckpt_dir, workdir)
    if config.eval_from.get('step') is None:
        ckpt_list = ckpt_manager.unevaluated_checkpoints(
            timeout=config.eval_from.get('timeout', 3600 * 8))
    elif config.eval_from.step > 0:
        # Evaluates at a specified step
        ckpt_list = [
            c for c in ckpt_manager.unevaluated_checkpoints(0, return_all=True)
            if ckpt_manager.sort_key_fn(c) == config.eval_from.step
        ]
        assert ckpt_list, f'Checkpoint at step {config.eval_from.step} not found.'
    elif config.eval_from.step == -1:  # The last checkpoint.
        ckpt_list = [*ckpt_manager.unevaluated_checkpoints(0, return_all=True)][-1:]
        assert ckpt_list, 'No checkpoint found.'
    elif config.eval_from.step == 0:
        ckpt_list = [ckpt_dir]  # For unit test.
    else:
        raise ValueError(f'Invalid step to evaluate: {config.eval_from.step}')
    return config, ckpt_manager, ckpt_list


def sample_step(train_state: TrainState, batch: Batch, *,
                model_dict: Dict[str, nn.Module],
                config: ml_collections.ConfigDict):
    """Runs a single step to generate samples given input videos.

    Args:
      train_state: The state of training TrainState
      batch: A single batch of data. Dictionary where
        batch['inputs'].shape= device_bs, t, h, w, 3
        batch['batch_mask'].shape= device_bs
        batch['batch_mask'].dtype= float32
        where batch['batch_mask'] > 0 for valid examples
      model_dict: A dictionary of generator and discriminator Flax models
      config: Configurations of the experiment.

    Returns:
      Sampled videos formatted in spatial grids.
    """
    variables = {
        'params': train_state.g_params,
        **train_state.g_model_state
    }
    ema_variables = {
        'params': train_state.ema_params,
        **train_state.g_model_state
    }
    if config.vqgan.model_type == '2D' and config.get('dataset_type', 'video') == 'image':
        batch['inputs'] = batch['inputs'][:, None]
    generator = model_dict['generator']
    # generate_fn = functools.partial(generator.apply, variables)
    # ema_generate_fn = functools.partial(generator.apply, ema_variables)
    outputs = {
        'original_video': batch['inputs'],
    }
    if 'batch_mask' in batch:
        outputs.update(dict(batch_mask=batch['batch_mask']))

    if 'label' in batch:
        outputs['label'] = batch['label']
    # generated_video, _ = generate_fn(batch['inputs'])
    # outputs['generated_video'] = jnp.clip(generated_video, 0, 1)
    # generated_video_ema, result_dict_ema = ema_generate_fn(batch['inputs'])
    # outputs['generated_video_ema'] = jnp.clip(generated_video_ema, 0, 1)
    # outputs['generated_tokens_ema'] = result_dict_ema['encoding_indices']
    results = {}

    encode_fn = functools.partial(generator.apply, variables, method=generator.encode)
    decoder_fn = functools.partial(generator.apply, variables, method=generator.decode)
    quantized, result_dict_ema = encode_fn(batch['inputs'])
    generated_video = decoder_fn(quantized)
    outputs['generated_video'] = jnp.clip(generated_video, 0, 1)

    ema_encode_fn = functools.partial(generator.apply, ema_variables, method=generator.encode)
    ema_decoder_fn = functools.partial(generator.apply, ema_variables, method=generator.decode)
    # decoder_fn = functools.partial(generator.decode, ema_variables)
    quantized, result_dict_ema = ema_encode_fn(batch['inputs'])
    generated_video_ema = ema_decoder_fn(quantized)
    outputs['generated_video_ema'] = jnp.clip(generated_video_ema, 0, 1)
    outputs['generated_tokens_ema'] = result_dict_ema['encoding_indices']
    print("777" * 10)
    print(quantized.shape)
    print(quantized)

    return outputs, results


def eval_step_get_features(
        outputs: Dict[str, Any], *,
        metric_params: Dict[str, Any],
        model_suffix_list: Iterable[str] = ('',),
        config: ml_collections.ConfigDict) -> EvalFeatureDict:
    """Extract features to compute eval metrics.

    Args:
      outputs: Dict of original video, generated video and batch_mask.
      metric_params: Params for metric models.
      model_suffix_list: Tuple of model suffixes, '' for the default and '_ema'
        for the ema model.
      config: Configurations of the experiment.

    Returns:
      Extracted metric features.
    """
    features = {'batch_mask': outputs['batch_mask']}
    size_suffixes = ['']
    if config.eval.get('image_resize') is not None:
        suffix = f'_resize{config.eval.image_resize}'
        for key in [*outputs.keys()]:
            if 'video' in key:
                outputs[f'{key}{suffix}'] = metrics_lib.resize_bilinear(
                    outputs[key], (config.eval.image_resize, config.eval.image_resize))
        size_suffixes.append(suffix)
        if config.eval.get('resized_only'):
            # Only evaluate resized samples, not original ones.
            size_suffixes.remove('')
    if config.eval.get('enable_inception_score', False):
        for s1, s2 in itertools.product(model_suffix_list, size_suffixes):
            suffix = s1 + s2
            features[f'inception_feature{suffix}'] = inception_score.run_model(
                metric_params['inception_score'],
                outputs[f'generated_video{suffix}'])
    if config.eval.get('enable_frechet_distance', False):
        for suffix in size_suffixes:
            if f'original_video{suffix}' not in outputs:
                continue
            features[f'frechet_feature_orig{suffix}'] = frechet_distance.run_model(
                metric_params['frechet_distance'], outputs[f'original_video{suffix}'],
                **config.eval.get('frechet_distance_args', {}))
        for s1, s2 in itertools.product(model_suffix_list, size_suffixes):
            suffix = s1 + s2
            if f'generated_video{suffix}' not in outputs:
                continue
            features[f'frechet_feature{suffix}'] = frechet_distance.run_model(
                metric_params['frechet_distance'],
                outputs[f'generated_video{suffix}'],
                **config.eval.get('frechet_distance_args', {}))
    if config.eval.get('enable_ssim_psnr', False):
        for s1, s2 in itertools.product(model_suffix_list, size_suffixes):
            suffix = s1 + s2
            cur_out_dict = image_quality_metrics.run_models(
                outputs[f'original_video{s2}'],
                outputs[f'generated_video{suffix}'],
                is_tf_function=False,
                metric_functions=None)
            features.update(
                {f'{k}{suffix}': v for k, v in cur_out_dict.items()})
            cur_out_dict = image_quality_metrics.run_models(
                outputs[f'generated_video{suffix}'][:, 1:],
                outputs[f'generated_video{suffix}'][:, :-1],
                is_tf_function=False,
                metric_functions=None)
            features.update(
                {f'interframe_{k}{suffix}': v for k, v in cur_out_dict.items()})
    if config.eval.get('enable_utilization', False):
        for suffix in model_suffix_list:
            if f'generated_tokens{suffix}' not in outputs:
                continue
            tokens = outputs[f'generated_tokens{suffix}']
            tokens = jax.tree_util.tree_map(
                lambda x: x.reshape(x.shape[0], -1), tokens)
            tokens = jnp.concatenate(jax.tree_leaves(tokens), axis=-1)
            features[f'tokens{suffix}'] = tokens

    features = jax.lax.all_gather(features, axis_name='device', tiled=True)
    return features  # total_batch_size, ...


def eval_step(
        train_state: TrainState,
        batch: Batch,
        *,
        model_dict: Dict[str, nn.Module],
        config: ml_collections.ConfigDict,
        metric_params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Runs a single step of evaluation.

    Args:
      train_state: The state of evaluating TrainState
      batch: A single batch of data. Dictionary where batch['inputs'].shape=
        device_bs, t, h, w, 3
      model_dict: A dictionary of generator and discriminator Flax models
      config: Configurations of the experiment.
      metric_params: Params for metric models.

    Returns:
      Metric features and generated outputs.
    """
    outputs, _ = sample_step(
        train_state, batch, model_dict=model_dict, config=config)

    dataset_type = config.get('dataset_type', 'video')
    if config.vqgan.model_type == '2D' and dataset_type == 'image':
        batch['inputs'] = batch['inputs'][:, None]
    if dataset_type == 'image':
        # Extend to 16 frames for IS and FVD.
        batch_mask = outputs.pop('batch_mask', None)
        tokens = outputs.pop('generated_tokens_ema')
        outputs = jax.tree_util.tree_map(lambda x: jnp.tile(x, (1, 16, 1, 1, 1)),
                                         outputs)
        outputs['generated_tokens_ema'] = tokens
        if batch_mask is not None:
            outputs['batch_mask'] = batch_mask

    features = eval_step_get_features(
        outputs,
        metric_params=metric_params,
        model_suffix_list=['', '_ema'],
        config=config)

    if dataset_type == 'image':
        # Reduce back to 1 frame.
        batch_mask = outputs.pop('batch_mask', None)
        tokens = outputs.pop('generated_tokens_ema')
        outputs = jax.tree_util.tree_map(lambda x: x[:, :1], outputs)
        outputs['generated_tokens_ema'] = tokens
        if batch_mask is not None:
            outputs['batch_mask'] = batch_mask

    if config.eval.get('enable_lpips', False):
        outputs = jax.lax.all_gather(outputs, axis_name='device', tiled=True)
    return features, outputs


def eval_step_encode(
        train_state: TrainState,
        batch: Batch,
        *,
        model_dict: Dict[str, nn.Module],
        config: ml_collections.ConfigDict,
        metric_params: Dict[str, Any]) -> jnp.ndarray:
    variables = {
        'params': train_state.g_params,
        **train_state.g_model_state
    }
    ema_variables = {
        'params': train_state.ema_params,
        **train_state.g_model_state
    }
    if config.vqgan.model_type == '2D' and config.get('dataset_type', 'video') == 'image':
        batch['inputs'] = batch['inputs'][:, None]
    generator = model_dict['generator']
    # generate_fn = functools.partial(generator.apply, variables)
    # ema_generate_fn = functools.partial(generator.apply, ema_variables)
    outputs = {
        'original_video': batch['inputs'],
    }
    if 'batch_mask' in batch:
        outputs.update(dict(batch_mask=batch['batch_mask']))

    if 'label' in batch:
        outputs['label'] = batch['label']
    encode_fn = functools.partial(generator.apply, variables, method=generator.encode)
    decoder_fn = functools.partial(generator.apply, variables, method=generator.decode)
    quantized, result_dict_ema = encode_fn(batch['inputs'])
    generated_video = decoder_fn(quantized)
    outputs['generated_video'] = jnp.clip(generated_video, 0, 1)

    ema_encode_fn = functools.partial(generator.apply, ema_variables, method=generator.encode)
    quantized, result_dict_ema = ema_encode_fn(batch['inputs'])
    return quantized


def eval_step_decode(
        train_state: TrainState,
        batch: Batch,
        quantized: jnp.ndarray,
        *,
        model_dict: Dict[str, nn.Module],
        config: ml_collections.ConfigDict,
        metric_params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    variables = {
        'params': train_state.g_params,
        **train_state.g_model_state
    }
    ema_variables = {
        'params': train_state.ema_params,
        **train_state.g_model_state
    }
    if config.vqgan.model_type == '2D' and config.get('dataset_type', 'video') == 'image':
        batch['inputs'] = batch['inputs'][:, None]
    generator = model_dict['generator']
    # generate_fn = functools.partial(generator.apply, variables)
    # ema_generate_fn = functools.partial(generator.apply, ema_variables)
    outputs = {
        'original_video': batch['inputs'],
    }
    if 'batch_mask' in batch:
        outputs.update(dict(batch_mask=batch['batch_mask']))

    if 'label' in batch:
        outputs['label'] = batch['label']
    # generated_video, _ = generate_fn(batch['inputs'])
    # outputs['generated_video'] = jnp.clip(generated_video, 0, 1)
    # generated_video_ema, result_dict_ema = ema_generate_fn(batch['inputs'])
    # outputs['generated_video_ema'] = jnp.clip(generated_video_ema, 0, 1)
    # outputs['generated_tokens_ema'] = result_dict_ema['encoding_indices']
    results = {}

    decoder_fn = functools.partial(generator.apply, variables, method=generator.decode)
    generated_video = decoder_fn(quantized)
    outputs['generated_video'] = jnp.clip(generated_video, 0, 1)

    ema_decoder_fn = functools.partial(generator.apply, ema_variables, method=generator.decode)
    generated_video_ema = ema_decoder_fn(quantized)
    outputs['generated_video_ema'] = jnp.clip(generated_video_ema, 0, 1)
    outputs['generated_tokens_ema'] = quantized

    dataset_type = config.get('dataset_type', 'video')
    if config.vqgan.model_type == '2D' and dataset_type == 'image':
        batch['inputs'] = batch['inputs'][:, None]
    if dataset_type == 'image':
        # Extend to 16 frames for IS and FVD.
        batch_mask = outputs.pop('batch_mask', None)
        tokens = outputs.pop('generated_tokens_ema')
        outputs = jax.tree_util.tree_map(lambda x: jnp.tile(x, (1, 16, 1, 1, 1)),
                                         outputs)
        outputs['generated_tokens_ema'] = tokens
        if batch_mask is not None:
            outputs['batch_mask'] = batch_mask

    features = eval_step_get_features(
        outputs,
        metric_params=metric_params,
        model_suffix_list=['', '_ema'],
        config=config)

    if dataset_type == 'image':
        # Reduce back to 1 frame.
        batch_mask = outputs.pop('batch_mask', None)
        tokens = outputs.pop('generated_tokens_ema')
        outputs = jax.tree_util.tree_map(lambda x: x[:, :1], outputs)
        outputs['generated_tokens_ema'] = tokens
        if batch_mask is not None:
            outputs['batch_mask'] = batch_mask

    if config.eval.get('enable_lpips', False):
        outputs = jax.lax.all_gather(outputs, axis_name='device', tiled=True)
    return features, outputs


def evaluate_encode(*, rng: jnp.ndarray, config: ml_collections.ConfigDict, input_spec: list,
                    batch: Batch, workdir: str):
    """Main evaluation loop lives in this function."""
    config, _, ckpt_list = get_eval_jobs(workdir, config)
    # Build the flax_model and the optimizers.
    _, init_rng = jax.random.split(rng)
    model_dict, train_state, _ = create_train_state(input_spec, config, init_rng,
                                                    False)

    eval_step_encode_pmapped = jax.pmap(
        functools.partial(
            eval_step_encode,
            model_dict=model_dict,
            config=config,
            metric_params=None
        ),
        axis_name='device',
        # We can donate the buffer of batch.
        donate_argnums=(1),
    )

    result_dir = None
    if config.eval.get('results_dir') is not None:
        result_dir = config.eval.results_dir
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)

    print("333" * 10)
    print(ckpt_list)
    for ckpt_path in ckpt_list:
        # Restores the model
        if not gfile.exists(ckpt_path):
            logging.warn(
                'Unable to evaluate ckpt %s because it does not exist. '
                'If this is a parallel evaluation job, try to increase '
                'config.logging.checkpoint_kept or use more accelerators.', ckpt_path)
            continue
        print("start load ckpt")
        train_state = train_utils.restore_checkpoint(
            ckpt_path,
            train_state,
            is_legacy_checkpoint=config.eval_from.get('legacy_checkpoint', False))

        train_state_replicated = jax_utils.replicate(train_state)
        eval_batch = batch
        print("555" * 10)
        print("start encode")
        quantized = eval_step_encode_pmapped(
            train_state_replicated, eval_batch)
        quantized = jax.device_get(quantized)

        # save quantized
        quantized = quantized.copy()
        tokenname = f'result.npz'
        path = os.path.join(result_dir, tokenname)
        np.savez(path, data=quantized)

        del train_state_replicated

    return path


def evaluate_decode(*, rng: jnp.ndarray, config: ml_collections.ConfigDict,input_spec: list,
                    batch: Batch, quantized: jnp.ndarray, workdir: str):
    config, _, ckpt_list = get_eval_jobs(workdir, config)
    # Build the flax_model and the optimizers.
    _, init_rng = jax.random.split(rng)
    model_dict, train_state, _ = create_train_state(input_spec, config, init_rng,
                                                    False)

    eval_step_decode_pmapped = jax.pmap(
        functools.partial(
            eval_step_decode,
            model_dict=model_dict,
            config=config,
            metric_params=None
        ),
        axis_name='device',
        # We can donate the buffer of batch.
        donate_argnums=(1),
    )

    label_names, result_dir, write_executor = None, None, None
    if config.eval.get('results_dir') is not None:
        result_dir = config.eval.results_dir
        write_executor = concurrent.futures.ThreadPoolExecutor(100)

    all_metrics, batch_samples = {}, None
    print("333" * 10)
    for ckpt_path in ckpt_list:
        # Restores the model
        if not gfile.exists(ckpt_path):
            logging.warn(
                'Unable to evaluate ckpt %s because it does not exist. '
                'If this is a parallel evaluation job, try to increase '
                'config.logging.checkpoint_kept or use more accelerators.', ckpt_path)
            continue
        print("start load ckpt")
        train_state = train_utils.restore_checkpoint(
            ckpt_path,
            train_state,
            is_legacy_checkpoint=config.eval_from.get('legacy_checkpoint', False))

        train_state_replicated = jax_utils.replicate(train_state)
        # eval_batch = next(dataset.valid_iter)
        eval_batch = batch
        print("666" * 10)
        print("start decode")
        _, batch_outputs = eval_step_decode_pmapped(
            train_state_replicated, eval_batch, quantized)
        # batch_features = jax.device_get(jax_utils.unreplicate(batch_features))
        # eval_features.append(batch_features)

        batch_outputs = jax.device_get(batch_outputs)

        batch_samples = jax.tree_util.tree_map(
            lambda x: x.reshape(-1, *x.shape[2:]), batch_outputs)
        if write_executor is not None:
            print("666" * 10)
            if 'generated_video_ema' in batch_samples:  # For VQ models
                videos = batch_samples['generated_video_ema']
                print(videos.shape)
                if config.eval.get('results_with_condition', True):
                    video = np.concatenate((videos, batch_samples['original_video']), axis=2)
                    # imagename = f'result.png'
                    # path = os.path.join(result_dir, imagename)
                    # write_executor.submit(mediapy.write_image, path, video[0][0])
                    print(video.shape)
                    filename = f'result.mp4'
                    path = os.path.join(result_dir, filename)
                    write_executor.submit(mediapy.write_video, path, video[0], fps=8)

        del train_state_replicated

    return train_state, all_metrics, batch_samples


def evaluate(*, rng: jnp.ndarray, config: ml_collections.ConfigDict,
             dataset: dataset_utils.Dataset, workdir: str,
             writer: metric_writers.MetricWriter):
    """Main evaluation loop lives in this function."""
    lead_host = jax.process_index() == 0
    dtype = train_utils.get_dtype(config)
    config, _, ckpt_list = get_eval_jobs(workdir, config)
    # Build the flax_model and the optimizers.
    _, init_rng = jax.random.split(rng)
    input_spec = [(
        dataset.meta_data['input_shape'],  # bs, t, h, w, 3
        dataset.meta_data.get('input_dtype', dtype))]
    model_dict, train_state, _ = create_train_state(input_spec, config, init_rng,
                                                    False)

    # Eval step pmap.
    # eval_step_pmapped = jax.pmap(
    #     functools.partial(
    #         eval_step,
    #         model_dict=model_dict,
    #         config=config,
    #         metric_params=None
    #     ),
    #     axis_name='device',
    #     # We can donate the buffer of batch.
    #     donate_argnums=(1),
    # )

    eval_step_encode_pmapped = jax.pmap(
        functools.partial(
            eval_step_encode,
            model_dict=model_dict,
            config=config,
            metric_params=None
        ),
        axis_name='device',
        # We can donate the buffer of batch.
        donate_argnums=(1),
    )

    eval_step_decode_pmapped = jax.pmap(
        functools.partial(
            eval_step_decode,
            model_dict=model_dict,
            config=config,
            metric_params=None
        ),
        axis_name='device',
        # We can donate the buffer of batch.
        donate_argnums=(1),
    )

    label_names, result_dir, write_executor = None, None, None
    if config.eval.get('results_dir') is not None:
        result_dir = config.eval.results_dir
        write_executor = concurrent.futures.ThreadPoolExecutor(100)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)

    all_metrics, batch_samples = {}, None
    print("333" * 10)
    for ckpt_path in ckpt_list:
        # Restores the model
        if not gfile.exists(ckpt_path):
            logging.warn(
                'Unable to evaluate ckpt %s because it does not exist. '
                'If this is a parallel evaluation job, try to increase '
                'config.logging.checkpoint_kept or use more accelerators.', ckpt_path)
            continue

        train_state = train_utils.restore_checkpoint(
            ckpt_path,
            train_state,
            is_legacy_checkpoint=config.eval_from.get('legacy_checkpoint', False))

        train_state_replicated = jax_utils.replicate(train_state)
        eval_batch = next(dataset.valid_iter)
        print("555" * 10)

        # _, batch_outputs = eval_step_pmapped(
        #     train_state_replicated, eval_batch)
        quantized = eval_step_encode_pmapped(
            train_state_replicated, eval_batch)
        quantized = jax.device_get(quantized)

        _, batch_outputs = eval_step_decode_pmapped(
            train_state_replicated, eval_batch, quantized)
        # batch_features = jax.device_get(jax_utils.unreplicate(batch_features))
        # eval_features.append(batch_features)

        batch_outputs = jax.device_get(batch_outputs)

        batch_samples = jax.tree_util.tree_map(
            lambda x: x.reshape(-1, *x.shape[2:]), batch_outputs)
        if write_executor is not None:
            print("666" * 10)
            # batch_samples is local flattened batch as np.array
            # batch_features is global flattened batch as np.array
            # batch_features = get_local_batch(batch_features)
            # batch_samples.update(batch_features)
            if 'generated_video_ema' in batch_samples:  # For VQ models
                videos = batch_samples['generated_video_ema']
                print(videos.shape)
                if config.eval.get('results_with_condition', True):
                    video = np.concatenate((videos, batch_samples['original_video']), axis=2)
                    # imagename = f'result.png'
                    # path = os.path.join(result_dir, imagename)
                    # write_executor.submit(mediapy.write_image, path, video[0][0])
                    print(video.shape)
                    filename = f'result.mp4'
                    path = os.path.join(result_dir, filename)
                    write_executor.submit(mediapy.write_video, path, video, fps=8)

        del train_state_replicated

    return train_state, all_metrics, batch_samples


def evaluate_model(config, dataset, workdir, writer):
    # 应用模型进行编码
    rng = jax.random.PRNGKey(0)
    return evaluate(rng=rng, config=config, dataset=dataset, workdir=workdir, writer=writer)


if __name__ == '__main__':
    # 使用tfds加载数据集
    config = vqgan3d_custom_dataset_config_infer_eval.get_config()  # 示例配置，您需要替换为您的配置
    writer = metric_writers.create_default_writer(
        "./writer_dir", just_logging=jax.process_index() > 0, asynchronous=True)
    data_rng = jax.random.PRNGKey(0)
    dataset = train_utils.get_dataset(
        config,
        data_rng,
        False)
    dtype = train_utils.get_dtype(config)
    input_spec = [(
        dataset.meta_data['input_shape'],  # bs, t, h, w, 3
        dataset.meta_data.get('input_dtype', dtype))]
    eval_batch = next(dataset.valid_iter)

    # eval_batch={}
    # image = cv2.imread('/workspace/eval_image/magvit/videogvt/models/256.jpg')
    # image = cv2.resize(image, (128, 128))
    # # 将 BGR 图像转换为 RGB 图像
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #
    # # 创建一个形状为 (1, 1, 256, 256, 3) 的空数组
    # output_array = np.empty((1, 17, 128, 128, 3), dtype=np.float32)
    #
    # # 将转换后的 RGB 图像添加到输出数组中
    # output_array[0, :, :, :, :] = [image_rgb / 255.0 for i in range(17)]
    # array = jnp.array(output_array)
    # # array = jnp.array(np.random.uniform(0, 1, size=(1, 1, 256, 256, 3)))
    # # 缩放到 0 到 1 之间
    # eval_batch['inputs'] = array
    # eval_batch['batch_mask'] = jnp.array(np.array([[1.0]], dtype=np.float32))
    # eval_batch['label'] = jnp.array(np.array([[1.0]], dtype=np.int32))

    workdir = '/workspace/v2/magvit/videogvt/dir00'
    rng = jax.random.PRNGKey(0)
    token_path = evaluate_encode(rng=rng, config=config, input_spec=input_spec, batch=eval_batch, workdir=workdir)
    quantized = None
    # 读取 npz 文件
    with np.load(token_path) as data:
        # 获取名为 'data' 的 NumPy 数组
        quantized = data['data']
    quantized = jnp.array(quantized)
    evaluate_decode(rng=rng, config=config, input_spec=input_spec, batch=eval_batch,quantized=quantized, workdir=workdir)
    # evaluate_model(config, dataset, workdir, writer)

