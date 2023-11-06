from torch.utils.data import Dataset
from PIL import Image
import os


class MyDataset(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.image_path = os.listdir(self.path)

    def __getitem__(self, index):
        # 读取图片名字
        image_name = self.image_path[index]
        # 获取图片地址
        image_item_path = os.path.join(self.path, image_name)
        image = Image.open(image_item_path)
        # 获取图片标签
        label = self.label_dir
        return image, label

    def __len__(self):
        return len(self.image_path)
