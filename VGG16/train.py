# plot绘图参考：https://blog.csdn.net/u013950379/article/details/87936999

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms
import matplotlib.pyplot as plt
import time

# 自定义import
from Net.Net import Net
from LoadData.Dataset import ReadData

def train(model, train_dataset, batch_size, lr):
    global loss
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4) ##shuffle得开，自己数据集是有规律的，bro
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    size = len(train_loader.dataset)
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        # 计算误差
        pred = model(X)
        loss = loss_fn(pred, y).to(device)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    print("Finished training")
    return loss

def val(model, val_dataset, batch_size):
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)  ##shuffle得开，自己数据集是有规律的，bro
    loss_fn = nn.CrossEntropyLoss()
    size = len(val_loader.dataset)
    num_batches = len(val_dataset)
    # 关闭批量归一化
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(val_loader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    val_loss /= num_batches
    correct /= size
    torch.save(model.state_dict(), './Model/VGG16_model' + str(100*correct) + '.pth')
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f} ")
    return val_loss

if __name__ == '__main__':
    ax = []  # 定义一个 x 轴的空列表用来接收动态的数据
    ay = []
    bx = []
    by = []
    plt.ion()  # 开启一个画图的窗口进入交互模式，用于实时更新数据
    # plt.rcParams['savefig.dpi'] = 200 #图片像素
    # plt.rcParams['figure.dpi'] = 200 #分辨率
    plt.rcParams['figure.figsize'] = (10, 10)  # 图像显示大小
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 防止中文标签乱码，还有通过导入字体文件的方法
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['lines.linewidth'] = 0.5  # 设置曲线线条宽度

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    # 加载模型到GPU
    model = Net().to(device)

    # transform整理数据集，参见官方Doc，比自己写好使多了！但是如果考虑模型精度问题，还是自己调整可能会更好
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize([32, 32]),
        torchvision.transforms.ToTensor(),
        # 现象：未开启Normalize前CPU占用很高，开启后占用低了，训练速度也没下降，？
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 只在tensor工作（标准化加快模型收敛）
        #torchvision.transforms.CenterCrop(150),
    ])
    # 使用自定义类读取数据集
    data = ReadData(file_name='E:/Program/DL/data/train_labels.csv', transforms=transforms)
    # 数据集划分，数值可更改(划分为训练集和验证集)
    train_size = int(0.95 * len(data))
    val_size = len(data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(data, [train_size, val_size])

    epochs = 100
    batch_size = 32
    lr = 0.001

    for epoch in range(epochs):
        t1 = time.time()
        print(f"Epoch {epoch + 1}-------------------------------")
        train_loss = train(model, train_dataset, batch_size, lr)
        val_loss = val(model, val_dataset, batch_size)
        print(f"total used time:{(time.time() - t1):>0.1f}\n")

        # 画图画图/看看loss的变化
        plt.clf()  # 清除刷新前的图表，防止数据量过大消耗内存
        plt.suptitle("Loss", fontsize=30)  # 添加总标题，并设置文字大小
        # 图表1
        train_loss = train_loss.cpu().detach().numpy()
        ax.append(epoch + 1)  # 追加x坐标值
        ay.append(train_loss)  # 追加y坐标值
        agraphic = plt.subplot(2, 1, 1)
        agraphic.set_title('Train_loss')  # 添加子标题
        agraphic.set_xlabel('Epoch', fontsize=10)  # 添加轴标签
        agraphic.set_ylabel('Loss', fontsize=20)
        plt.plot(ax, ay, 'g-')  # 等于agraghic.plot(ax,ay,'g-')
        # 图表2
        bx.append(epoch + 1)
        by.append(val_loss)
        bgraghic = plt.subplot(2, 1, 2)
        bgraghic.set_title('Val_loss')
        bgraghic.plot(bx, by, 'r^')
        plt.pause(0.5)  # 设置暂停时间，太快图表无法正常显示
    plt.ioff()
    plt.show()