from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
# transfrom指的是transform工具箱，其中包含了很多对图像进行变换的函数
# 最常用的是以下三个函数
# transforms.Compose()函数则是将多个变换组合在一起
# transforms.ToTensor()函数则是将图像转换为Tensor类型
# transforms.Resize()函数则是对图像进行缩放

# 类不接受任何一个参数，必须要写实例化类，给实例化后的类传递参数
image_path = "data/train/bees/1092977343_cb42b38d62.jpg"
writer = SummaryWriter("logs")

img_PIL = Image.open(image_path)
# 实例化类
To_tensor = transforms.ToTensor()
# 给类传递参数
img_tensor = To_tensor(img_PIL)
# print(img_tensor)
writer.add_image("Tensor", img_tensor, 1, dataformats="CHW")

writer.close()
# transforms.Normalize()函数则是对图像进行归一化处理
# transforms.RandomCrop()函数则是对图像进行随机裁剪
# transforms.RandomHorizontalFlip()函数则是对图像进行随机水平翻转
# transforms.RandomVerticalFlip()函数则是对图像进行随机垂直翻转
# transforms.CenterCrop()函数则是对图像进行中心裁剪
# transforms.Pad()函数则是对图像进行填充
# transforms.Grayscale()函数则是对图像进行灰度处理
# transforms.RandomRotation()函数则是对图像进行随机旋转
# transforms.ColorJitter()函数则是对图像进行颜色变换

