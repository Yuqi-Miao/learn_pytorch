# 可以使用tensorboard实现可视化网络结构
import torch
from torch import nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms as T


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.ReLU(x)
        return x


dataset_transform = T.Compose([
    T.ToTensor(),
])
myNet = MyNet()
test_set = torchvision.datasets.CIFAR10('./dataset', train=False, transform=dataset_transform, download=True)
data_loader = DataLoader(test_set, batch_size=64)
writer = SummaryWriter('./log-4')
step = 0
for data in data_loader:
    img, label = data
    writer.add_images('input', img, step)
    img_out = myNet(img)
    writer.add_images('output', img_out, step)
    step += 1

writer.close()
