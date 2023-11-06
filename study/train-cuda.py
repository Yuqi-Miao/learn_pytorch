# 找三个部分 1.网络模型 2.数据（输入+标注） 3.损失函数
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
from model import *
from torch.utils.tensorboard import SummaryWriter
# import time

# data pre
train_data = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
# data loader
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# 创建网络模型
mynet = MyNet()
# 调用cuda
mynet = mynet.cuda()
# 定义损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
# 定义优化器
optimizer = torch.optim.Adam(mynet.parameters(), lr=0.01)
# 训练
total_train_step = 0
total_test_step = 0
epoch = 10

# tensorboard
writer = SummaryWriter("./log-train")
# start_time = time.time()
for i in range(epoch):
    print("------第{}轮训练开始------".format(i + 1))
    # 训练开始
    mynet.train()
    for data in train_dataloader:
        img, label = data
        # 调用cuda
        img = img.cuda()
        label = label.cuda()
        outputs = mynet(img)
        loss = loss_fn(outputs, label)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            # end_time = time.time()
            # print(end_time-start_time)
            print("训练次数：{}, Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试开始
    # 每次的loss重新设置为0
    total_test_loss = 0
    # torch.agrmax()返回最大值的索引 1表示按行取最大值 0表示按列取最大值 返回的是索引 也就是类别 与label进行比较 得到正确率 也就是准确率 accuracy
    total_accuracy = 0
    mynet.eval()
    with torch.no_grad():
        for data in test_dataloader:
            img, label = data
            img = img.cuda()
            label = label.cuda()
            outputs = mynet(img)
            loss = loss_fn(outputs, label)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == label).sum()
            total_accuracy += accuracy
    total_test_step += 1
    print("在测试集商整体的Loss:{}".format(total_test_loss))
    print("在测试集上的准确率:{}".format(total_accuracy.item() / len(test_data)))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / len(test_data), total_test_step)

# 保存每一轮的模型
#    torch.save(mynet.state_dict(), "./model/model_{}.pth".format(i + 1))

writer.close()
# # len
# print(len(train_data))
# print(len(test_data))
