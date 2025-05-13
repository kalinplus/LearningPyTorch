import os
from torch.utils.data import Dataset
import cv2
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.img_list = os.listdir(os.path.join(root_dir, label_dir))

    def __getitem__(self, index):
        img_name = self.img_list[index]
        img_path = os.path.join(self.root_dir, self.label_dir, img_name)
        image = Image.open(img_path)
        label = self.label_dir
        return image, label
    def __len__(self):
        return len(self.img_list)

root_dir = "请指定昆虫数据集路径"
ants_label_dir = "ants"
bees_label_dir = "bees"

ant_ds = MyDataset(root_dir, label_dir)
bee_ds = MyDataset(root_dir, bees_label_dir)

# 可以直接对数据集进行拼接，常用于数据增强、构造子数据集、构造 mock 数据集等
train_ds = ant_ds + bee_ds