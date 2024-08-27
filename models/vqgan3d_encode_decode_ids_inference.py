import concurrent
import functools
import itertools
import logging

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
DEFAULT_HEIGHT = 128
DEFAULT_WIDTH = 128
DEFAULT_CHANNELS = 3
DEFAULT_LENGTH = 17


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


def eval_step_encode(
        train_state: TrainState,
        batch: Batch,
        *,
        model_dict: Dict[str, nn.Module],
        config: ml_collections.ConfigDict,
        metric_params: Dict[str, Any]) -> jnp.ndarray:

    ema_variables = {
        'params': train_state.ema_params,
        **train_state.g_model_state
    }
    if config.vqgan.model_type == '2D' and config.get('dataset_type', 'video') == 'image':
        batch['inputs'] = batch['inputs'][:, None]
    generator = model_dict['generator']

    ema_encode_fn = functools.partial(generator.apply, ema_variables, method=generator.encode_to_indices)
    ids = ema_encode_fn(batch['inputs'])
    return ids


def eval_step_decode(
        train_state: TrainState,
        batch: Batch,
        quantized: jnp.ndarray,
        *,
        model_dict: Dict[str, nn.Module],
        config: ml_collections.ConfigDict,
        metric_params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ema_variables = {
        'params': train_state.ema_params,
        **train_state.g_model_state
    }
    if config.vqgan.model_type == '2D' and config.get('dataset_type', 'video') == 'image':
        batch['inputs'] = batch['inputs'][:, None]
    generator = model_dict['generator']

    outputs = {
        'original_video': batch['inputs'],
    }

    ema_decoder_fn = functools.partial(generator.apply, ema_variables, method=generator.decode_from_indices)
    generated_video_ema = ema_decoder_fn(quantized)
    outputs['generated_video_ema'] = jnp.clip(generated_video_ema, 0, 1)
    outputs['generated_tokens_ema'] = quantized

    dataset_type = config.get('dataset_type', 'video')
    if config.vqgan.model_type == '2D' and dataset_type == 'image':
        batch['inputs'] = batch['inputs'][:, None]
    if dataset_type == 'image':
        # Extend to 16 frames for IS and FVD.
        tokens = outputs.pop('generated_tokens_ema')
        outputs = jax.tree_util.tree_map(lambda x: jnp.tile(x, (1, 16, 1, 1, 1)),
                                         outputs)
        outputs['generated_tokens_ema'] = tokens

    if dataset_type == 'image':
        # Reduce back to 1 frame.
        tokens = outputs.pop('generated_tokens_ema')
        outputs = jax.tree_util.tree_map(lambda x: x[:, :1], outputs)
        outputs['generated_tokens_ema'] = tokens

    if config.eval.get('enable_lpips', False):
        outputs = jax.lax.all_gather(outputs, axis_name='device', tiled=True)
    return outputs


def create_model(*, rng: jnp.ndarray, config: ml_collections.ConfigDict, input_spec: list, workdir: str) -> Tuple[TrainState, Dict[str, nn.Module]]:
    """Create the model and the optimizer."""
    config, _, ckpt_list = get_eval_jobs(workdir, config)
    # Build the flax_model and the optimizers.
    _, init_rng = jax.random.split(rng)
    model_dict, train_state, _ = create_train_state(input_spec, config, init_rng,
                                                    False)
    print("333" * 10)
    print(ckpt_list)
    ckpt_path = ckpt_list[0]
    # Restores the model
    if not gfile.exists(ckpt_path):
        logging.warn(
            'Unable to evaluate ckpt %s because it does not exist. '
            'If this is a parallel evaluation job, try to increase '
            'config.logging.checkpoint_kept or use more accelerators.', ckpt_path)
        return
    print("start load ckpt")
    train_state = train_utils.restore_checkpoint(
        ckpt_path,
        train_state,
        is_legacy_checkpoint=config.eval_from.get('legacy_checkpoint', False))

    train_state_replicated = jax_utils.replicate(train_state)

    return train_state_replicated, model_dict

def evaluate_encode(*, config: ml_collections.ConfigDict, train_state_replicated: TrainState, model_dict: Dict[str, nn.Module],
                    batch: Batch, token_save_path: str):
    """Main evaluation loop lives in this function."""

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

    eval_batch = batch
    print("555" * 10)
    print("start encode")
    quantized = eval_step_encode_pmapped(
        train_state_replicated, eval_batch)
    quantized = jax.device_get(quantized)

    # save quantized
    quantized = quantized.copy()
    np.save(token_save_path, quantized)
    print(quantized)
    print("save quantized", quantized.shape)

    del train_state_replicated


def evaluate_decode(*, config: ml_collections.ConfigDict, train_state_replicated: TrainState, model_dict: Dict[str, nn.Module],
                    quantized: jnp.ndarray, workdir: str):


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

    # label_names, result_dir, write_executor = None, None, None
    # if config.eval.get('results_dir') is not None:
    #     result_dir = config.eval.results_dir
    #     write_executor = concurrent.futures.ThreadPoolExecutor(100)
    #
    # all_metrics, batch_samples = {}, None


    print("666" * 10)
    print("start decode")
    batch_outputs = eval_step_decode_pmapped(
        train_state_replicated, eval_batch, quantized)


    batch_outputs = jax.device_get(batch_outputs)

    batch_samples = jax.tree_util.tree_map(
        lambda x: x.reshape(-1, *x.shape[2:]), batch_outputs)
    # if write_executor is not None:
    #     print("666" * 10)
    #     if 'generated_video_ema' in batch_samples:  # For VQ models
    #         videos = batch_samples['generated_video_ema']
    #         print(videos.shape)
    #         if config.eval.get('results_with_condition', True):
    #             video = np.concatenate((videos, batch_samples['original_video']), axis=2)
    #             # imagename = f'result.png'
    #             # path = os.path.join(result_dir, imagename)
    #             # write_executor.submit(mediapy.write_image, path, video[0][0])
    #             print(video.shape)
    #             filename = f'result.mp4'
    #             path = os.path.join(result_dir, filename)
    #             mediapy.write_video(path, video[0], fps=8)
    #             # write_executor.submit(mediapy.write_video, path, video[0], fps=8)

    del train_state_replicated
    return batch_samples['generated_video_ema']


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
    # [((-1, 17, 128, 128, 3), <class 'jax.numpy.float32'>)]
    # input_spec = [(
    #     dataset.meta_data['input_shape'],  # bs, t, h, w, 3
    #     dataset.meta_data.get('input_dtype', dtype))]
    test_mode = "video"
    t = 17
    if test_mode == "image":
        t = 1
    input_spec = [(
        (-1, t, 128, 128, 3),  # bs, t, h, w, 3
        jax.numpy.float32)]
    print(input_spec)
    eval_batch = next(dataset.valid_iter)
    eval_batch = next(dataset.valid_iter)
    eval_batch = next(dataset.valid_iter)
    print(input_spec)
    if test_mode == "image":
        eval_batch['inputs'] = eval_batch['inputs'][:,:,:1,:,:,:]  # 图片
    print(eval_batch['inputs'].shape)

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

    workdir = '/workspace/v2/magvit/videogvt/dir01'
    rng = jax.random.PRNGKey(0)
    create_model = functools.partial(create_model, rng=rng, config=config, input_spec=input_spec, workdir=workdir)
    train_state_replicated, model_dict = create_model()
    tokenname = f'result.npy'
    result_dir = None
    if config.eval.get('results_dir') is not None:
        result_dir = config.eval.results_dir
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)
    token_path = os.path.join(result_dir, tokenname)
    evaluate_encode(config=config, train_state_replicated=train_state_replicated, model_dict=model_dict, batch=eval_batch, token_save_path=token_path)
    quantized = None
    # 读取 npz 文件
    token_path = '/workspace/v2/magvit/videogvt/models/target_path/43830/video_tokens.npy'
    quantized = np.load(token_path)
    print(quantized)
    print(quantized.shape)
    # with np.load(token_path) as data:
    #     # 获取名为 'data' 的 NumPy 数组
    #     quantized = data['data']
    #     print(quantized)
    #     print(quantized.shape)
    quantized = jnp.array(quantized)
    batch_samples = evaluate_decode(config=config, train_state_replicated=train_state_replicated, model_dict=model_dict, quantized=quantized, workdir=workdir)

    print("666" * 10)
    print(batch_samples)
    print(batch_samples.shape)
    if config.eval.get('results_dir') is not None:
        result_dir = config.eval.results_dir
        write_executor = concurrent.futures.ThreadPoolExecutor(100)
    if write_executor is not None:
        print("666" * 10)
        videos = batch_samples
        print(videos.shape)
        print(eval_batch['inputs'].shape)
        if config.eval.get('results_with_condition', True):
            video = np.concatenate((videos, eval_batch['inputs'][0]), axis=2)
            # imagename = f'result.png'
            # path = os.path.join(result_dir, imagename)
            # write_executor.submit(mediapy.write_image, path, video[0][0])
            print(video.shape)
            filename = f'result.mp4'
            path = os.path.join(result_dir, filename)
            mediapy.write_video(path, video[0], fps=8)
            # write_executor.submit(mediapy.write_video, path, video[0], fps=8)

