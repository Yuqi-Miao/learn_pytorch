from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

image_path = "data/train/bees/1092977343_cb42b38d62.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

writer = SummaryWriter('logs')
# 写入图片 “test”代表一个表的标题，img_array是图片的数据，2代表第二个图，HWC是数据格式
writer.add_image("test", img_array, 2, dataformats='HWC')
# y=x
# for i in range(100):
#     writer.add_scalar("y=2x", 2*i, i)
# 打开tensorboard命令：tensorboard --logdir=logs --port=6007
# http://localhost:6007/
writer.close()