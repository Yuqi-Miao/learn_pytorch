from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# 1.直接print 2.print(type()) 3.使用dir() 4.使用help()        
writer = SummaryWriter("logs")
img = Image.open('image/0013035.jpg')
# print(img)
trans_to_tensor = transforms.ToTensor()
img_tensor = trans_to_tensor(img)
# print(img_tensor)
writer.add_image("Tensor", img_tensor, 1, dataformats='CHW')
# Normalization 均值 标准差
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
writer.add_image("Normalize", img_norm, 1, dataformats='CHW')
# Compose
trans_resize = transforms.Resize((256, 256))
trans_compose = transforms.Compose([trans_resize, trans_to_tensor])
img_resize = trans_compose(img)
writer.add_image("Compose", img_resize, 0, dataformats='CHW')
# trans_resize = transforms.Resize((256, 256))
# trans_compose = transforms.Compose([trans_to_tensor,trans_resize])
# img_resize = trans_compose(img)
# writer.add_image("Compose-1", img_resize, 0, dataformats='CHW')

trans_random = transforms.RandomCrop((256, 256))
trans_compose_2 = transforms.Compose([trans_random, trans_to_tensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("Compose-2", img_crop, i, dataformats='CHW')

writer.close()
