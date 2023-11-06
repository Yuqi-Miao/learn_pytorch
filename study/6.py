import torchvision
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset_transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = torchvision.datasets.CIFAR10('./dataset', train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10('./dataset', train=False, transform=dataset_transform, download=True)

data_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0, drop_last=True)
writer = SummaryWriter('./log-3')
for epoch in range(2):
    step = 0
    for data in data_loader:
        img, target = data
        writer.add_images('test_set_{}'.format(epoch), img, step)
        step = step + 1
        # print(img.shape)
        # print(target)

#
# for i in range(10):
#     img, target = test_set[i]
#     writer.add_image('test_set', test_set[i][0], i)

writer.close()
# print(train_set)
# print(train_set.classes)
