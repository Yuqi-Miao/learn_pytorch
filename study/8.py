import torchvision
import torch

Vgg16_false = torchvision.models.vgg16(pretrained=False)


Vgg16_false.classifier[6] = torch.nn.Linear(in_features=4096, out_features=10, bias=True)
print(Vgg16_false)
# Vgg16_true = torchvision.models.vgg16(pretrained=True, progress=True)
#
# print(Vgg16_true)
