from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, excel_file, transform=None):
        self.excel_file = excel_file
        self.data = pd.read_excel(excel_file)
        self.images_path = pd.read_excel(excel_file)['path'].tolist()
        self.images_class = pd.read_excel(excel_file)['label'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item].replace('/media/ubuntu/8TDisk/sz/data_new/', 'xxxxx/data_new/'))
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

class MyDataSet3D(Dataset):
    """自定义数据集"""

    def __init__(self, excel_file, transform=None):
        self.excel_file = excel_file
        self.data = pd.read_excel(excel_file)
        self.videos_path = pd.read_excel(excel_file)['path'].tolist()
        # self.videos_class = pd.read_excel(excel_file)['label1'].tolist()
        self.videos_class = pd.read_excel(excel_file)['label2'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.videos_path)

    def __getitem__(self, item):
        video = np.load(self.videos_path[item])
        # video = np.load(self.videos_path[item].replace('/media/ubuntu/8TDisk/乳腺/', '/media/ubuntu/8TDisk/乳腺/Breast/'))
        # RGB为彩色图片，L为灰度图片
        # if img.mode != 'RGB':
        #     raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.videos_class[item]

        if self.transform is not None:
            processed_frames = []
            for frame in video:
                frame = Image.fromarray(frame)  # 将 NumPy 数组转换为 PIL 图像
                frame = self.transform(frame)  # 应用其他转换
                processed_frames.append(frame)

            video = torch.stack(processed_frames)  # 将帧列表转换为张量

        return video, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        videos, labels = tuple(zip(*batch))

        videos = torch.stack(videos, dim=0)
        labels = torch.as_tensor(labels)
        return videos, labels

class MyDataSet2D(Dataset):
    """自定义数据集"""

    def __init__(self, excel_file, transform=None):
        self.excel_file = excel_file
        self.data = pd.read_excel(excel_file)
        self.images_path = pd.read_excel(excel_file)['path'].tolist()
        # self.images_class = pd.read_excel(excel_file)['label'].tolist()
        self.images_class = pd.read_excel(excel_file)['label2'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        video = np.load(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        # if img.mode != 'RGB':
        #     raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]
        image = video[8]
        if self.transform is not None:
            image = Image.fromarray(image)
            image = self.transform(image)  # 应用其他转换

        return image, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels