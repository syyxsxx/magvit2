# https://www.tensorflow.org/datasets/add_dataset?hl=zh-cn
# https://github.com/tensorflow/datasets/issues/5355
import os
import random
import time
import traceback
import logging

import cv2
import tensorflow_datasets as tfds
import tensorflow as tf

# tf.config.run_functions_eagerly(True)
class CustomDataset(tfds.core.GeneratorBasedBuilder):
    """Builder for my custom dataset."""
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    DEFAULT_HEIGHT = 128
    DEFAULT_WIDTH = 128
    DEFAULT_CHANNELS = 3
    DEFAULT_LENGTH = 17
    DEFAULT_TRAIN_PERCENTAGE = 0.8
    DEFAULT_DATA_DIR = './a/'
    NUM_TRAIN_EXAMPLES = 400000
    NUM_TEST_EXAMPLES = 100000

    def __init__(self, data_dir=None, dataset_path='', **kwargs):
        super(CustomDataset, self).__init__(
            data_dir=data_dir,
            **kwargs
        )
        self.dataset_path = dataset_path
        # Define your dataset specific attributes here.

    def _info(self):
        # 定义每个视频帧的特征
        frame_feature = tfds.features.Image(shape=(self.DEFAULT_HEIGHT, self.DEFAULT_WIDTH, self.DEFAULT_CHANNELS), encoding_format='jpeg')
        # 使用 Sequence 来表示一个视频的所有帧
        video_feature = tfds.features.Sequence(frame_feature, length=self.DEFAULT_LENGTH)

        features = tfds.features.FeaturesDict({
            'video': video_feature,
            'label': tfds.features.ClassLabel(num_classes=2),
        })
        return tfds.core.DatasetInfo(
            builder=self,
            description='A custom video dataset.',
            features=features,
            homepage='',
        )

    def _split_generators(self, dl_manager):
        # 假设 dataset_dir 是您的数据集目录
        dataset_dir = self.dataset_path
        # self.info.splits['train'].num_examples = self.NUM_TRAIN_EXAMPLES
        # self.info.splits['test'].num_examples = self.NUM_TEST_EXAMPLES
        # 定义数据集的分割
        datasets = []
        dataset_dir = "/app/train_data/video_infos/"
        for video_id in list(sorted([int(i) for i in os.listdir(dataset_dir)])):
            path = '{}/{}'.format(dataset_dir, video_id)
            datasets.append(path)
        # dataset_dir = "/app/train_data/heads_with_audio/"
        # for video_id in list(sorted([int(i) for i in os.listdir(dataset_dir)])):
        #     path = '{}/{}'.format(dataset_dir, video_id)
        #     datasets.append(path)
        random.seed(42)
        datasets = random.sample(datasets, len(datasets))
        return {
            'train': self._generate_examples(datasets, 'train'),
            'test': self._generate_examples(datasets, 'test'),
        }

    def _generate_examples(self, datasets, split):
        # 生成每个分割的示例
        trainsets_len = int(len(datasets) * self.DEFAULT_TRAIN_PERCENTAGE)
        if split == 'train':
            # 选择xxx个子文件夹作为训练集
            selected_folders = datasets[:self.NUM_TRAIN_EXAMPLES]
        elif split == 'test':
            # 选择xxx个子文件夹作为测试集
            selected_folders = datasets[self.NUM_TRAIN_EXAMPLES:self.NUM_TRAIN_EXAMPLES+self.NUM_TEST_EXAMPLES]

        start = time.time()
        for i, folder in enumerate(selected_folders):
            # 假设每个子文件夹包含16张图片
            logging.info('Processing folder %s', folder)
            image_names = [image_file for image_file in os.listdir(folder) if
                      image_file.endswith('.jpg') or image_file.endswith('.png')]
            image_names = sorted(image_names, key=lambda x: float(x.split('f')[1].split('_')[0]))
            images = [os.path.join(folder, image_file) for image_file in image_names]
            try:
                for image_path in images:
                    # 读取图像并检查其大小
                    resize_image(image_path, self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)
            except Exception as e:
                print("Error in folder {}".format(folder))
                print(e)
                continue
            print("[{}] {}/{}".format(split, i, len(selected_folders)))
            if i>0 and i % 10000 == 0:
                cost_t = time.time() - start
                print("run {} Time {} remaining time {}".format(i, cost_t, (len(selected_folders)-i)*(cost_t/i)))
            # print(images)
            if len(images) == 0:
                print("No images in folder {}".format(folder))
                continue
            if len(images) < self.DEFAULT_LENGTH:
                print("Not enough images in folder {} {}".format(folder, len(images)))
                images = images + [images[-1]] * (self.DEFAULT_LENGTH - len(images))
            if len(images) == self.DEFAULT_LENGTH:
                yield folder, {'video': images, 'label': 1}


def resize_image(image_path, target_width=128, target_height=128):
    # 读取图片
    image = cv2.imread(image_path)
    # # 检查图片是否为空
    # if image is None:
    #     print(f"Could not read image: {image_path}")
    #     return
    # 获取图片原始大小
    original_width, original_height = image.shape[:2]
    # 判断图片大小是否为128x128
    if original_width != target_width or original_height != target_height:
        print(f"Resizing image: {image_path} {(original_width, original_height)}")
        # 调整图片大小到128x128
        image = cv2.resize(image, (target_width, target_height))
        # 保存调整大小的图片
        cv2.imwrite(image_path, image)


if __name__ == "__main__":
    try:
        # Now you can use your builder like this:
        builder = CustomDataset(dataset_path="/workspace/custom_dataload/custom_dataset/")
        # builder = tfds.builder("custom_dataset")
        builder.download_and_prepare()
        dataset = builder.as_dataset(split='train',batch_size=2)

        dataset_length = dataset.reduce(
            initial_state=0,
            reduce_func=lambda state, _: state + 1,
        )

        print(f"Dataset length: {dataset_length}")

        print(builder.info.splits['train'].num_examples)

        for example in dataset:
            video = example['video']
            label = example['label']
            # print(video, label)
            print(video.shape, label.shape)
            break

        dataset_test = builder.as_dataset(split='test', batch_size=2)
        dataset_length = dataset_test.reduce(
            initial_state=0,
            reduce_func=lambda state, _: state + 1,
        )

        print(f"Dataset label length: {dataset_length}")

        print(builder.info.splits['test'].num_examples)

        for example in dataset_test:
            video = example['video']
            label = example['label']
            # print(video, label)
            print(video.shape, label.shape)
            break

    except Exception as e:
        # Handle error
        traceback.print_exc()
        pass
