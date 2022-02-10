# 测试pred可视化, test.py没写，知道原理大抵和前面train，val差不多，重写就完了，注意没有grad
# Pred参考CSDN博主文章：https://blog.csdn.net/qq_44807176/article/details/112640863
import torch
from torchvision import transforms
from PIL import Image,ImageDraw,ImageFont
import csv
import cv2
import numpy as np

# 自定义import
from Net.Net import Net

def pred(path):
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 标准化,训练开启了标准化的
         ])

    classes = ('cat',
                'dog')

    net = Net().to(device)  # 实例化骨架
    net.load_state_dict(torch.load('Model/GoogleNet_model86.8.pth'))  # 载入权重文件

    # 载入图像
    pic = Image.open(path)

    img = transform(pic)  # [C, H, W]预处理成pytorch能处理的tensor格式
    img = torch.unsqueeze(img, dim=0)  # [N, C, H, W]加上一个新维度batch
    img = img.to(device)
    with torch.no_grad():
        outputs = net(img)
        predict = torch.softmax(outputs, dim=1)  # 对第一个维度进行处理
    print(predict)
    pred = classes[predict[0].argmax(0)]
    print(pred)

    # PIL的方法添加文字也用过，但是没有找到好的方法切图，用进程的方法太麻烦，所以暂用熟悉的cv2
    pic = cv2.cvtColor(np.asarray(pic), cv2.COLOR_RGB2BGR)
    AddText = pic.copy()
    cv2.putText(AddText, pred, (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)

    # 将原图片和添加文字后的图片拼接起来
    cv2.imshow("OpenCV", AddText)
    k = cv2.waitKey(0)  # waitkey代表读取键盘的输入，括号里的数字代表等待多长时间，单位ms。 0代表一直等待
    if k == 27:  # 键盘上Esc键的键值
        cv2.destroyAllWindows()

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    #path = 'E:/Program/DL/data/test/609.jpg'
    # pred(path)
    filename = "E://Program//DL//data//test_labels.csv"
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            path = row['file_name']
            pred(path)