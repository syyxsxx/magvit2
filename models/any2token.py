import functools
import logging
import os
import sys
import time
import argparse

import cv2
import numpy as np
import urllib3
import jax
import jax.numpy as jnp
from multiprocessing import Pool

import requests
import traceback
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, DateTime, Boolean, func, or_, not_
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from videogvt.configs import vqgan3d_custom_dataset_config_infer_eval
from videogvt.train_lib import train_utils
from videogvt.models.vqgan3d_encode_decode_ids_inference import create_model, evaluate_encode

urllib3.disable_warnings()
# 配置日志记录器
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

source_path = '/workspace/v2/magvit/videogvt/models/source_path'
target_path = '/workspace/v2/magvit/videogvt/models/target_path'
image_clip_len = 17
connect_args = {
    'connect_timeout': 60,  # 设置连接超时为10秒
}
engine = create_engine('postgresql://admin:U6ox`co-rYGLP%f~@35.193.231.176:5432/prompts', connect_args=connect_args)
Base = declarative_base()


class VideoInfos(Base):
    __tablename__ = "video_infos"
    id = Column(Integer, primary_key=True)
    url = Column(String(512))
    source_url = Column(String(512))
    caption = Column(String(1024))
    opticalFlow_score = Column(Float)
    aesthetic_score = Column(Float)
    ocr_score = Column(Float)
    image_text_match_score = Column(Float)
    source = Column(String(128))
    create_at = Column(DateTime, default=func.now())
    status = Column(String(128))
    _pass = Column('pass', Boolean)
    limite_condition = Column(String(128))
    duration = Column(Float)
    width = Column(Integer)
    height = Column(Integer)
    fps = Column(Float)
    total_frame = Column(Integer)
    memo = Column(String(128))
    camera_motion = Column(String(128))
    review = Column(Boolean, default=False)
    file_size = Column(Float)
    has_audio = Column(Boolean)
    caption_image_url = Column(String(512))
    local_path = Column(String(512))
    local_image_path = Column(String(512))
    seaweedfs_url = Column(String(512))
    seaweedfs_caption_img_url = Column(String(512))
    instance_name = Column(String(128))
    has_embedding = Column(String(128))
    get_kf_done = Column(Boolean, default=False)
    caption_after_process = Column(String(2048))
    kf_128x128 = Column(String(4096))


def download_file(url, file_name):
    try:
        # 发送 GET 请求下载视频文件
        headers = {}
        if 'seaweedfs' in url:
            headers["Authorization"] = "Basic aW1hZ2luZWFwcDpDcmVhdG9ybmZ0czEh"
        response = requests.get(url, stream=True, headers=headers, timeout=5, verify=False)
        if not str(response.status_code).startswith('2'):
            # 请求失败，状态码不以2开头
            print('download failed: {}'.format(url))
            raise Exception('download failed: {}'.format(url))

        # 保存视频文件到本地
        with open(file_name, 'wb') as f:
            f.write(response.content)

        return file_name
    except TimeoutError as e:
        raise Exception('download failed: {} timeout'.format(url))


def download_images(image_url, save_path):
    try:
        # 发送 GET 请求下载视频文件
        headers = {}
        if 'seaweedfs' in image_url:
            headers["Authorization"] = "Basic aW1hZ2luZWFwcDpDcmVhdG9ybmZ0czEh"
        response = requests.get(image_url, stream=True, headers=headers, timeout=5, verify=False)
        if not str(response.status_code).startswith('2'):
            # 请求失败，状态码不以2开头
            print('download failed: {}'.format(image_url))
            raise Exception('download failed: {}'.format(image_url))

        # 保存视频文件到本地
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        image_file = '{}/{}'.format(save_path, image_url.split('/')[-1])
        with open(image_file, 'wb') as f:
            f.write(response.content)

        return image_file
    except TimeoutError as e:
        raise Exception('download failed: {} timeout'.format(image_url))


def create_video_model(test_mode = "video"):
    config = vqgan3d_custom_dataset_config_infer_eval.get_config()  # 示例配置，您需要替换为您的配置
    t = 17
    if test_mode == "image":
        t = 1
    input_spec = [(
        (-1, t, 128, 128, 3),  # bs, t, h, w, 3
        jax.numpy.float32)]
    print(input_spec)

    rng = jax.random.PRNGKey(0)
    workdir = '/workspace/v2/magvit/dir01'
    create_train_model = functools.partial(create_model, rng=rng, config=config, input_spec=input_spec, workdir=workdir)
    train_state_replicated, model_dict = create_train_model()
    return train_state_replicated, model_dict, config


def process_video_infos(video_paths, config, train_state_replicated, model_dict):
    l = len(video_paths)
    for index, video_path in enumerate(video_paths):
        try:
            video_id = video_path.split('/')[-1]
            image_names = os.listdir(os.path.join(source_path, str(video_id), "images"))
            if len(image_names) != image_clip_len:
                print("images not enough! all data {}".format(image_names))
                continue
            print('=============start video token {} {}/{}!================'.format(video_id, index, l))
            token_path = os.path.join(target_path, str(video_id), "video_tokens.npy")
            if os.path.exists(token_path):
                print("video token exists! data id {}".format(video_id))
                continue
            if not os.path.exists(os.path.join(target_path, str(video_id))):
                os.makedirs(os.path.join(target_path, str(video_id)), exist_ok=True)

            eval_batch={}

            image_names = sorted(image_names, key=lambda x: float(x.split('/')[-1].split('_')[-2].split('f')[-1]))
            # print(image_names)
            image_paths = [os.path.join(os.path.join(source_path, str(video_id), "images"), image_name) for image_name in
                           image_names]
            images = []
            for image_path in image_paths:
                image = cv2.imread(image_path)
                image = cv2.resize(image, (128, 128))
                # 将 BGR 图像转换为 RGB 图像
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
                images.append(image_rgb)
            images = np.array(images)
            # 创建一个形状为 (1, 1, 256, 256, 3) 的空数组作为输入
            output_array = np.empty((1, 1, 17, 128, 128, 3), dtype=np.float32)
            # print("images.shape", images.shape)
            # 将转换后的 RGB 图像添加到输出数组中
            output_array[0, :, :, :, :, :] = images
            print("output_array.shape", output_array.shape)
            array = jnp.array(output_array)
            # array = jnp.array(np.random.uniform(0, 1, size=(1, 1, 256, 256, 3)))
            # 缩放到 0 到 1 之间
            eval_batch['inputs'] = array
            eval_batch['batch_mask'] = jnp.array(np.array([[1.0]], dtype=np.float32))
            eval_batch['label'] = jnp.array(np.array([[1.0]], dtype=np.int32))

            evaluate_encode(config=config, train_state_replicated=train_state_replicated, model_dict=model_dict,
                            batch=eval_batch, token_save_path=token_path)
        except Exception as e:
            print('=============process video token failed!================')
            traceback.print_exc()
            print("process video token failed! data id {}".format(video_id))
            continue


def process_image_info(video_paths, config, train_state_replicated, model_dict):
    l = len(video_paths)
    for index, video_path in enumerate(video_paths):
        try:
            video_id = video_path.split('/')[-1]
            image_names = os.listdir(os.path.join(source_path, str(video_id), "images"))
            if len(image_names) != image_clip_len:
                print("images not enough! all data {}".format(image_names))
                continue
            print('=============start image token {} {}/{}!================'.format(video_id, index, l))
            token_path = os.path.join(target_path, str(video_id), "image_token.npy")
            if os.path.exists(token_path):
                print("video token exists! data id {}".format(video_id))
                continue
            if not os.path.exists(os.path.join(target_path, str(video_id))):
                os.makedirs(os.path.join(target_path, str(video_id)), exist_ok=True)

            eval_batch={}
            image_names = os.listdir(os.path.join(source_path, str(video_id), "images"))
            image_names = sorted(image_names, key=lambda x: float(x.split('/')[-1].split('_')[-2].split('f')[-1]))
            image_paths = [os.path.join(os.path.join(source_path, str(video_id), "images"), image_name) for image_name in image_names]
            # print(image_paths, image_paths[int(len(image_paths)/2)])
            image = cv2.imread(image_paths[int(len(image_paths)/2)])
            image = cv2.resize(image, (128, 128))
            # 将 BGR 图像转换为 RGB 图像
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            image = np.array(image_rgb)

            # 创建一个形状为 (1, 1, 256, 256, 3) 的空数组
            output_array = np.empty((1, 1, 1, 128, 128, 3), dtype=np.float32)

            # 将转换后的 RGB 图像添加到输出数组中
            output_array[0, :, :, :, :, :] = image
            array = jnp.array(output_array)
            # array = jnp.array(np.random.uniform(0, 1, size=(1, 1, 256, 256, 3)))
            # 缩放到 0 到 1 之间
            eval_batch['inputs'] = array
            eval_batch['batch_mask'] = jnp.array(np.array([[1.0]], dtype=np.float32))
            eval_batch['label'] = jnp.array(np.array([[1.0]], dtype=np.int32))

            evaluate_encode(config=config, train_state_replicated=train_state_replicated, model_dict=model_dict,
                            batch=eval_batch, token_save_path=token_path)
        except Exception as e:
            print('=============process image token failed!================')
            traceback.print_exc()
            print("process image token failed! data id {}".format(video_id))
            continue


def process_caption_info(video_paths):
    l = len(video_paths)
    for index, video_path in enumerate(video_paths):
        video_id = video_path.split('/')[-1]
        image_names = os.listdir(os.path.join(source_path, str(video_id), "images"))
        if len(image_names) != image_clip_len:
            print("images not enough! all data {}".format(image_names))
            continue
        print('=============start mv caption {} {}/{}!================'.format(video_id, index, l))
        txt_path = os.path.join(target_path, str(video_id), str(video_id)+".txt")
        if os.path.exists(txt_path):
            print("caption exists! data id {}".format(video_id))
            continue
        if not os.path.exists(os.path.join(target_path, str(video_id))):
            os.makedirs(os.path.join(target_path, str(video_id)), exist_ok=True)
        os.system("cp {} {}".format(os.path.join(source_path, str(video_id), str(video_id)+".txt"), txt_path))


def download_video_infos(video_info):
    try:
        current_id = video_info.id
        caption = video_info.caption_after_process
        save_image_path = os.path.join(source_path, str(current_id), "images")
        save_caption_path = os.path.join(source_path, str(current_id), str(current_id)+".txt")
        print('=============start download images {}!================'.format(current_id))
        url_list = video_info.kf_128x128.split(',')
        if not os.path.exists( os.path.join(source_path, str(current_id))):
            os.makedirs(os.path.join(source_path, str(current_id)), exist_ok=True)
        if not os.path.exists(save_image_path):
            os.makedirs(save_image_path, exist_ok=True)
            url_list = sorted(url_list, key=lambda x: float(x.split('/')[-1].split('_')[-2].split('f')[-1]))
            # print(url_list)
            for i, url in enumerate(url_list[4:-4]):
                download_images(url, save_image_path)

        with open(save_caption_path, 'w') as f:
            f.write(caption)

        l = len(list(os.listdir(save_image_path)))
        if l != image_clip_len:
            print("download images failed! all data {} {}".format(l, url_list))
            os.system("rm -rf {}".format(save_image_path))

    except Exception as e:
        print('=============download video failed!================')
        traceback.print_exc()
        print("download video failed! data id {}".format(current_id))


if __name__ == "__main__":
    # Session = sessionmaker(bind=engine)
    # session = Session()
    # # 分页查找get_kf_done=True的10w条数据
    # # 定义每页显示的数据条数
    # page_size = 20
    # # 计算总页数
    # # 假设我们不知道总数，但是我们知道我们想要查找到达100000条记录为止
    # current_id = 0  # 起始ID
    # start_id = current_id
    # total_records_needed = 10000
    # records_received = 0
    # # 遍历每一页，下载数据
    # while records_received < total_records_needed:
    #     try:
    #         start_time = time.time()
    #         video_infos = session.query(VideoInfos).filter(VideoInfos.id > current_id,
    #                                                        VideoInfos.caption_after_process.isnot(None),
    #                                                        VideoInfos.get_kf_done==True # ,VideoInfos.source=='UCF101'
    #                                                        ).order_by(VideoInfos.id).limit(page_size).all()
    #         # print("video_infos", video_infos)
    #         if not video_infos:
    #             break
    #
    #         # 使用多进程并行处理每个视频信息
    #         pool = Pool(processes=20)  # 使用所有可用CPU核心
    #         pool.map(download_video_infos, video_infos)
    #         pool.close()
    #         pool.join()
    #
    #         current_id = video_infos[-1].id
    #         records_received = len(list(os.listdir(source_path)))
    #         # 打印下载耗时
    #         cost_time = time.time() - start_time
    #         print('Download images cost time: {}s'.format(cost_time))
    #         print('============={}/{} start id {} download video cost time: {}m================'.format(
    #             records_received, total_records_needed, current_id,
    #             (total_records_needed - records_received) * cost_time / page_size / 60))
    #     except Exception as e:
    #         print('Download images failed!')
    #         print(e)
    #         traceback.print_exc()
    #         continue
    #     finally:
    #         session.close()

    parser = argparse.ArgumentParser(description='argparse token')
    parser.add_argument('--start_id', '-s', type=int, default=0, help='')
    parser.add_argument('--end_id', '-e', type=int, default=10000, help='')
    parser.add_argument('--clip_id', '-c', type=int, default=0, help='')
    args = parser.parse_args()
    start_id = args.start_id
    end_id = args.end_id
    source_names = sorted(list(os.listdir(source_path)), key=lambda x: int(x))
    source_paths = [os.path.join(source_path, video_id) for video_id in source_names if int(video_id)>=start_id and int(video_id)<end_id]
    clip_id = args.clip_id
    print("start_id", start_id, "end_id", end_id, "clip_id", clip_id, "source_paths len", len(source_paths))
    clip_start_id = len(source_paths)/10 * clip_id
    clip_end_id = len(source_paths)/10 * (clip_id+1)
    video_info_paths = source_paths[int(clip_start_id):int(clip_end_id)]
    print("source_paths len", len(source_paths), "video_info_paths len", len(video_info_paths), "start_id", video_info_paths[0], "end_id", video_info_paths[-1])
    # # # 计算视频的token
    train_state_replicated, model_dict, config = create_video_model(test_mode = "video")
    # video_info_paths = [os.path.join(source_path, video_id) for video_id in os.listdir(source_path)[clip_start_id:clip_end_id]]
    process_video_infos(video_info_paths, config, train_state_replicated, model_dict)
    #
    # # 计算中间帧的token
    train_state_replicated, model_dict, config = create_video_model(test_mode = "image")
    # video_info_paths = [os.path.join(source_path, video_id) for video_id in os.listdir(source_path)[clip_start_id:clip_end_id]]
    process_image_info(video_info_paths, config, train_state_replicated, model_dict)
    #
    # # 移动caption
    # video_info_paths = [os.path.join(source_path, video_id) for video_id in os.listdir(source_path)[clip_start_id:clip_end_id]]
    process_caption_info(video_info_paths)

