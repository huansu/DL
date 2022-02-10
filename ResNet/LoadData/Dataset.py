import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image
import torchvision.transforms

# 加载数据,理解好这部分能够更清楚数据是怎么导入模型中的
class ReadData(Dataset):
    # 初始化
    def __init__(self, file_name=None, transforms=None, target_transform=None):
        # 读入数据
        self.data = pd.read_csv(file_name)
        self.target_transform = target_transform
        self.transforms = transforms

    # 返回df的长度
    def __len__(self):
        return len(self.data)

    # 获取第idx+1列的数据
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        image = read_image(img_path)  # 读入tensor类型
        label = self.data.iloc[idx, 1]
        if self.transforms:
            image = self.transforms(image)
        # 数据格式转换
        image = image.type(torch.FloatTensor)
        return image, label