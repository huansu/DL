# 数据预处理.py文件
# 对数据集进行整理，填入csv文件，和后面的数据Dataset读取数据直接挂钩
# ！！！更改数据集后需要重写代码使数据集符合要求

import os
import csv

filePath = 'E:/Program/DL/data/test/'
csv_file = open('E:/Program/DL/data/test_labels.csv','w',encoding='utf-8',newline="")
csv_write = csv.writer(csv_file)
csv_write.writerow(["file_name","label"])
names = os.listdir(filePath)
for name in names:
    img_path = filePath + str(name)
    if name.split(".")[0]=="cat":
        classes = 0
    else:
        classes = 0
    csv_write.writerow([img_path, classes])

csv_file.close()