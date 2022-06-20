from pathlib import Path
from typing import Tuple

import numpy as np
import torch as T
from PIL import Image
from torch.utils.data import Dataset


class ImageFolder720p(Dataset):
    """
    Image shape is (720, 1280, 3) --> (768, 1280, 3) --> 6x10 128x128 patches
    图像被填充到 1280x768（即 24,24 高度填充），以便它们可以分成 60 个 128x128 块。
    """

    def __init__(self, root: str):
        self.files = sorted(Path(root).iterdir())

    def __getitem__(self, index: int) -> Tuple[T.Tensor, np.ndarray, str]:
        path = str(self.files[index % len(self.files)])
        img = np.array(Image.open(path))
        # pad(array, pad_width, mode, **kwargs)数组填充 np.pad()
        pad = ((24, 24), (0, 0), (0, 0))

        # img = np.pad(img, pad, 'constant', constant_values=0) / 255
        img = np.pad(img, pad, mode="edge") / 255.0
        # np.transpose维度变化
        # 使用 numpy.transpose ()进行变换，
        # 其实就是交换了坐标轴，如：x.transpose(1, 2, 0)，
        # 其实就是将x第二维度挪到第一维上，第三维移到第二维上，原本的第一维移动到第三维上
        img = np.transpose(img, (2, 0, 1))
        img = T.from_numpy(img).float()

        patches = np.reshape(img, (3, 6, 128, 10, 128))
        patches = np.transpose(patches, (0, 1, 3, 2, 4))

        return img, patches, path

    def __len__(self):
        return len(self.files)
