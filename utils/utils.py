from torchvision import datasets, transforms
import torch
import os
from torch.utils.data import DataLoader
from typing import List


# 绘制混淆矩阵
def show_confusion_matrix(y_true: List[int or str], y_pre: List[int or str], labels: List[int or str]) -> None:
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots()
    # plt.figure(figsize=(3, 3))
    matrix = confusion_matrix(y_true, y_pre, labels=labels)

    sns.heatmap(
        matrix,
        annot=True,
        ax=ax,
        fmt='d',  # fmt表示标注的字符串格式
        linewidths=.5,  # 设置每个单元格之间的行宽
        cmap='RdPu'  # 色彩模式
    )

    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# 输出日志
def output_log(message: str) -> None:
    import logging
    logging.basicConfig(
        filename='log.log',
        filemode='w',
        level=logging.INFO,
        format='[%(levelname)s] %(asctime)s [%(filename)s:%(lineno)d]: %(message)s'
    )
    logging.info(message)


def see_model(model_path):
    import torch
    content = torch.load(model_path, map_location='cpu')
    print(content.keys())
    # print(content['key_name'])


def analysis(src_path: str) -> None:
    """
    输出数据的分布图

    :param src_path: 数据集地址
    :return: 数据分布图，横坐标类别，纵坐标数量
    """
    import os
    from matplotlib import pyplot as plt
    X, Y = [], []
    classes = os.listdir(src_path)
    num = 0
    for idx, cls in enumerate(classes):
        X.append(cls)
        cls_num = len(os.listdir(os.path.join(src_path, cls)))
        num += cls_num
        Y.append(cls_num)
    plt.figure(figsize=(20, 10))
    plt.title(f'total number = {num}')
    plt.bar(X, Y)
    plt.xticks(X)
    for x, y in zip(X, Y):  # 柱状图上标注数字
        plt.text(x, y, y, ha='center', va='bottom')  # ha表示水平对齐，va表示垂直对齐
    plt.xlabel('category')
    plt.ylabel('number')
    plt.show()


def get_csv(src_path: str, classes: List[str]) -> None:
    """
    :param src_path: 数据集所在目录
    :param classes: 类别列表
    :return: 获取数据集对应的csv文件，一列，列名path，每一行为文件路径
    """
    import os, csv
    for cls in classes:
        path = os.path.join(src_path, cls)
        with open(os.path.join(src_path, f'{cls}.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['path'])
            for file_name in os.listdir(path):
                writer.writerow([os.path.join(path, file_name)])
    print('Get csv successfully!')


def copySameFiles(src_path1: str, src_path2: str, dst_path: str) -> None:
    """
    从src_path1中选出和src_path2里同名的文件移至dst_path

    :param src_path1:
    :param src_path2:
    :param dst_path:
    :return:
    """
    from shutil import copy
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    file_list = os.listdir(src_path2)
    for file in file_list:
        copy(os.path.join(src_path1, file), dst_path)


def sampleFolders(src_path: str, dst_path: str, threshold: int) -> None:
    """
    从文件夹中随机抽样文件，文件名相同

    :param src_path: 数据集路径
    :param dst_path: 文件存储路径
    :param threshold: 抽样数量
    :return:
    """
    from random import sample
    from shutil import copy
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    file_list = os.listdir(src_path)
    for file in sample(file_list, min(len(file_list), threshold)):
        copy(os.path.join(src_path, file), dst_path)
    print('Sample successfully!')


def sample(src_path: str, dst_path: str, threshold: int) -> None:
    """
    数据集抽样

    :param threshold: 抽样数量
    :param src_path: 数据集存储路径
    :param dst_path: 抽样后的数据集保存路径
    :return:
    """
    from random import sample
    from shutil import copy
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    for cls in os.listdir(src_path):
        if not os.path.exists(os.path.join(dst_path, cls)):
            os.mkdir(os.path.join(dst_path, cls))
        file_list = os.listdir(os.path.join(src_path, cls))
        for file in sample(file_list, min(len(file_list), threshold)):
            copy(os.path.join(src_path, cls, file), os.path.join(dst_path, cls))
        print(f'Class {cls} sample successfully!')
    print('Sample successfully!')


# def sample(src_path: str, classes: List[str], threshold: int) -> None:
#     """
#     :param src_path: 数据集对应的csv文件所在目录
#     :param classes: 类别列表
#     :param threshold: 阈值
#     :return:
#     """
#     import os
#     import pandas as pd
#     for index, cls in enumerate(classes):
#         data = pd.read_csv(os.path.join(src_path, f'{cls}.csv'))
#         num_row = data.shape[0]
#         num = threshold if num_row >= threshold else num_row
#         data_sample = data.sample(num, axis=0)
#
#         data_sample.insert(data.shape[1], 'class', cls)  # 插入类别列
#         # print(data_sample)
#         data_sample.to_csv(os.path.join(src_path, 'sample.csv'), mode='a', index=False,
#                            header=1 if index == 0 else 0)  # mode=a表示可以追加写入数据,header=0表示不保留列名
#     print('Sample successfully!')
#
#
# def copy_sample_data(dst_path: str, classes: List[str]) -> None:
#     """
#     :param dst_path: 目标路径
#     :param classes: 类别列表
#     :return:
#     """
#     import os
#     import pandas as pd
#     from shutil import copy
#     if not os.path.exists(dst_path):
#         os.mkdir(dst_path)
#     for cls in classes:
#         path = os.path.join(dst_path, cls)
#         if not os.path.exists(path):
#             os.mkdir(path)
#     path = '/Users/zaizai/Downloads/16-Cls/sample.csv'  # 采样得到的csv文件路径
#     data = pd.read_csv(path)
#     for index, row in data.iterrows():  # 按行遍历
#         src = row['path']
#         copy(src, os.path.join(dst_path, f'{row["class"]}'))
#     print('Copy sample data successfully!')


def dataset_partition(src_path: str, dst_path: str, train_split_rate: float, val_split_rate: float,
                      test_split_rate: float) -> None:
    """
    :param src_path: 数据集存放地址
    :param dst_path: 划分好的数据集存放地址
    :param train_split_rate: 训练集划分比例
    :param val_split_rate: 验证集划分比例
    :param test_split_rate: 测试集划分比例
    :return: 划分好的数据集
    """
    import os, random, csv
    from shutil import move

    def make_dir(file_path: str) -> None:
        if os.path.exists(file_path):
            os.rmdir(file_path)
        os.makedirs(file_path)

    random.seed(0)
    data_class = os.listdir(src_path)  # listdir返回指定的路径下包含的文件或文件夹名字的列表

    print('The classes are:')
    print(data_class, len(data_class))  # 输出数据种类，数据种类默认为读取的文件夹的名称

    nums = {cls: len(os.listdir(os.path.join(src_path, cls))) for cls in data_class}  # 统计每一个类别下的样本数量

    def move_file(file_path: str, split_rate: float, kind: str) -> None:
        with open(os.path.join(dst_path, f'{kind}.csv'), 'w') as f:
            writer = csv.writer(f)
            for cls in data_class:
                make_dir(os.path.join(file_path, cls))
                num_class_path = os.path.join(src_path, cls)
                images = os.listdir(num_class_path)
                random.shuffle(images)
                selected = random.sample(images, k=int(nums[cls] * split_rate))  # 从序列中选择特定长度的项目列表

                for index, image in enumerate(images):
                    # 将划分好的数据复制到相应目录
                    if image in selected:
                        writer.writerow([image, cls])
                        data_image_path = os.path.join(num_class_path, image)
                        new_path = os.path.join(file_path, cls)
                        move(data_image_path, new_path)

    # 建立保存训练集的文件夹
    train_data_root = os.path.join(dst_path, 'train')
    make_dir(train_data_root)
    move_file(train_data_root, train_split_rate, 'train')

    # 建立保存验证集的文件夹
    val_data_root = os.path.join(dst_path, 'val')
    make_dir(val_data_root)
    move_file(val_data_root, val_split_rate, 'val')

    # 建立保存测试集的文件夹
    test_data_root = os.path.join(dst_path, 'test')
    move_file(test_data_root, test_split_rate, 'test')

    # rmtree(src_path)
    print('Partition successfully!')


def dataset_partition2(src_path: str, dst_path: str, train_split_rate: float, test_split_rate: float) -> None:
    """
    :param src_path: 数据集存放地址
    :param dst_path: 划分好的数据集存放地址
    :param train_split_rate: 训练集划分比例
    :param test_split_rate: 测试集划分比例
    :return: 划分好的数据集
    """
    import os, random, csv
    from shutil import move, rmtree, copy

    def make_file(file_path: str) -> None:
        if os.path.exists(file_path):
            os.rmdir(file_path)
        os.makedirs(file_path)

    random.seed(0)
    data_class = os.listdir(src_path)  # listdir返回指定的路径下包含的文件或文件夹名字的列表

    print('The classes are:')
    print(data_class, len(data_class))  # 输出数据种类，数据种类默认为读取的文件夹的名称

    nums = {cls: len(os.listdir(os.path.join(src_path, cls))) for cls in data_class}  # 统计每一个类别下的样本数量

    def move_file(file_path: str, split_rate: float, kind: str) -> None:
        with open(os.path.join(dst_path, f'{kind}.csv'), 'w') as f:
            writer = csv.writer(f)
            for cls in data_class:
                make_file(os.path.join(file_path, cls))
                num_class_path = os.path.join(src_path, cls)
                images = os.listdir(num_class_path)
                random.shuffle(images)
                selected = random.sample(images, k=int(nums[cls] * split_rate))  # 从序列中选择特定长度的项目列表

                for index, image in enumerate(images):
                    # 将划分好的数据复制到相应目录
                    if image in selected:
                        writer.writerow([image, cls])
                        data_image_path = os.path.join(num_class_path, image)
                        new_path = os.path.join(file_path, cls)
                        # move(data_image_path, new_path)
                        copy(data_image_path, new_path)

    # 建立保存训练集的文件夹
    train_data_root = os.path.join(dst_path, 'train')
    make_file(train_data_root)
    move_file(train_data_root, train_split_rate, 'train')

    # 建立保存测试集的文件夹
    test_data_root = os.path.join(dst_path, 'val')
    move_file(test_data_root, test_split_rate, 'val')

    # rmtree(src_path)
    print('Partition successfully!')


def dataset_partition3(src_path: str, dst_path: str, train_split_rate: float, val_split_rate: float,
                      test_split_rate: float) -> None:
    """
    :param src_path: 数据集存放地址
    :param dst_path: 划分好的数据集存放地址
    :param train_split_rate: 训练集划分比例
    :param val_split_rate: 验证集划分比例
    :param test_split_rate: 测试集划分比例
    :return: 划分好的数据集
    """
    import os, random, csv
    from shutil import move

    def make_dir(file_path: str) -> None:
        if os.path.exists(file_path):
            os.rmdir(file_path)
        os.makedirs(file_path)

    random.seed(0)
    data_class = os.listdir(src_path)  # listdir返回指定的路径下包含的文件或文件夹名字的列表

    print('The classes are:')
    print(data_class, len(data_class))  # 输出数据种类，数据种类默认为读取的文件夹的名称

    nums = {cls: len(os.listdir(os.path.join(src_path, cls))) for cls in data_class}  # 统计每一个类别下的样本数量

    def move_file(file_path: str, split_rate: float, kind: str) -> None:
        with open(os.path.join(dst_path, f'{kind}.csv'), 'w') as f:
            writer = csv.writer(f)
            for cls in data_class:
                make_dir(os.path.join(file_path, cls))
                num_class_path = os.path.join(src_path, cls)
                images = os.listdir(num_class_path)
                random.shuffle(images)
                selected = random.sample(images, k=int(nums[cls] * split_rate))  # 从序列中选择特定长度的项目列表

                for index, image in enumerate(images):
                    # 将划分好的数据复制到相应目录
                    if image in selected:
                        writer.writerow([image, cls])
                        data_image_path = os.path.join(num_class_path, image)
                        new_path = os.path.join(file_path, cls)
                        move(data_image_path, new_path)

    # 建立保存训练集的文件夹
    train_data_root = os.path.join(dst_path, 'train')
    make_dir(train_data_root)
    move_file(train_data_root, train_split_rate, 'train')

    # 建立保存验证集的文件夹
    val_data_root = os.path.join(dst_path, 'val')
    make_dir(val_data_root)
    move_file(val_data_root, val_split_rate, 'val')

    # 建立保存测试集的文件夹
    test_data_root = os.path.join(dst_path, 'test')
    move_file(test_data_root, test_split_rate, 'test')

    # rmtree(src_path)
    print('Partition successfully!')


def checkDataset(train_path: str, val_path: str, test_path: str) -> bool:
    """
    检查划分好的数据集是否相互独立

    :param train_path: 训练集路径
    :param val_path: 验证集路径
    :param test_path: 测试集路径
    :return: 相互独立返回True，反之False
    """
    import os
    from collections import Counter
    data_class = os.listdir(train_path)
    cnt = 0
    flag = True
    for cls in data_class:
        d1 = Counter(os.listdir(os.path.join(train_path, cls)))
        d2 = Counter(os.listdir(os.path.join(val_path, cls)))
        d3 = Counter(os.listdir(os.path.join(test_path, cls)))
        print(d1, d2, d3)
        for k, v in d1.items():
            if k in d2 or k in d3:
                print(os.path.join(train_path, cls, k))
                flag = False
                cnt += 1
        for k, v in d2.items():
            if k in d1 or k in d3:
                print(os.path.join(val_path, cls, k))
                flag = False
                cnt += 1
        for k, v in d3.items():
            if k in d1 or k in d2:
                print(os.path.join(test_path, cls, k))
                flag = False
                cnt += 1
    print(f'duplicate images {cnt}')
    return flag


def compareFolder(src_path1: str, src_path2: str) -> List[str]:
    """
    比较两个文件夹里的文件，删除并返回返回非共有的文件名列表

    :param src_path1: 文件夹路径1
    :param src_path2: 文件夹路径2
    :return: 文件名列表
    """
    import os
    from collections import Counter
    ans = []
    cnt1 = Counter(os.listdir(src_path1))
    cnt2 = Counter(os.listdir(src_path2))
    for k, v in cnt1.items():
        if cnt2[k] == 0:
            os.remove(os.path.join(src_path1, k))
            print(k)
            ans.append(k)
    for k, v in cnt2.items():
        if cnt1[k] == 0:
            os.remove(os.path.join(src_path2, k))
            print(k)
            ans.append(k)
    print(len(ans))
    return ans


def deleteAllSuffixFile(src_path: str, suf: str) -> None:
    """
    删除文件目录下以.suf为后缀的文件

    :param src_path: 目录路径
    :param suf: 后缀 pattern
    :return:
    """
    import os
    for cls in os.listdir(src_path):
        # print(cls)
        for file in os.listdir(os.path.join(src_path, cls)):
            # print(file)
            if file.find(suf) != -1:
                os.remove(os.path.join(src_path, cls, file))
    print('Delete Successfully!')


def mergeFolder(src_path1: str, src_path2: str, dst_path: str) -> None:
    """
    合并两个文件夹（类别一致），例如合并train和test

    :param src_path1: 路径1
    :param src_path2: 路径2
    :param dst_path: 合并后的存放路径
    :return:
    """
    import os
    from shutil import copy
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    def copy_file(src_path: str, dst_path: str) -> None:
        for cls in os.listdir(src_path):
            if not os.path.exists(os.path.join(dst_path, cls)):
                os.makedirs(os.path.join(dst_path, cls))
            if not os.path.isdir(os.path.join(dst_path, cls)):
                continue
            for file in os.listdir(os.path.join(src_path, cls)):
                copy(os.path.join(src_path, cls, file), os.path.join(dst_path, cls))
            print(f'Copy class {cls} Successfully!')

    copy_file(src_path1, dst_path)
    copy_file(src_path2, dst_path)
    print('Merge Successfully!')


def mergeCls(src_path: str, json_path: str, dst_path: str) -> None:
    """
    合并类，例如将16类合并成4类

    :param src_path: 存放数据集的路径
    :param json_path: 类别对应的json文件
    :param dst_path: 存放合并后的数据集路径
    :return:
    """
    import json, os
    from shutil import copy
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    with open(json_path, 'r') as f:
        mapping_dict = json.load(f)
        for cls, clsList in mapping_dict.items():
            if not os.path.exists(os.path.join(dst_path, cls)):
                os.mkdir(os.path.join(dst_path, cls))
            num = 0
            for src_cls in clsList:
                num += len(os.listdir(os.path.join(src_path, src_cls)))
                print(src_cls, len(os.listdir(os.path.join(src_path, src_cls))))
                for file in os.listdir(os.path.join(src_path, src_cls)):
                    copy(os.path.join(src_path, src_cls, file), os.path.join(dst_path, cls))
            print(f'Merge class {cls} successfully! Total number = {num}')


def judgeDifference(src_path1: str, src_path2: str) -> None:
    from collections import Counter
    cnt = Counter(os.listdir(src_path1) + os.listdir(src_path2))
    for k, v in cnt.items():
        if v > 1:
            print(k)


def findBrokenImage(src_path: str) -> None:
    import numpy as np
    from PIL import Image

    for image in os.listdir(src_path):
        try:
            img = Image.open(os.path.join(src_path, image))
        except IOError:
            print(os.path.join(src_path, image))
        try:
            img = np.asarray(img)
        except:
            print('corrupt img', os.path.join(src_path, image))


def compareFolder2(src_path1: str, src_path2: str) -> None:
    """
    比较连两个路径下是否有相同文件，并输出相同文件路径
    :param src_path1:
    :param src_path2:
    :return:
    """
    import collections
    cnt1, cnt2 = collections.Counter(), collections.Counter()
    for cls_list in os.listdir(src_path1):
        for cls in cls_list:
            for img in os.listdir(os.path.join(src_path1, cls)):
                cnt1[img] = 1
    for cls_list in os.listdir(src_path2):
        for cls in cls_list:
            for img in os.listdir(os.path.join(src_path2, cls)):
                cnt2[img] = 1
    cnt = 0
    for k, v in cnt1.items():
        if k in cnt2:
            cnt += 1
            print(k)
    for k, v in cnt2.items():
        if k in cnt1:
            cnt += 1
            print(k)
    print(f'duplicate images {cnt}')


def deleteDuplicatImages(src_path: str):
    """
    查找数据集中重复的图片并输出&删除
    :param src_path:
    :return:
    """
    import collections
    from os import remove
    cnt = collections.Counter()
    print(os.listdir(src_path))
    for cls in os.listdir(src_path):
        for img in os.listdir(os.path.join(src_path, cls)):
            cnt[img] += 1
    d_img = []
    num = 0
    for k, v in cnt.items():
        if v > 1:
            d_img.append(k)
            num += 1
    print(num)
    for cls in os.listdir(src_path):
        for img in d_img:
            if os.path.exists(os.path.join(src_path, cls, img)):
                remove(os.path.join(src_path, cls, img))
    print('Delete successfully!')

dataTrans = transforms.Compose([
    transforms.Resize([299, 299]),  # resize  长宽不一致
    #transforms.CenterCrop(224),  # 在图片的中间区域进行裁剪
    transforms.ToTensor(),  # 将PIL image转成(c,h,w)的tensor格式，并除255进行归一化处理
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 逐通道进行标准化，均值变为0，标准差变为1
])

def read_data(data_dir, batch_size):
    train_data_dir = f'{data_dir}/train'
    val_data_dir = f'{data_dir}/val'
    # test_data_dir = f'{data_dir}/test'
    # 这里可自行修改数据预处理，batch大小也可自行调整
    # 保持本地训练的数据读取和这里一致
    train_dataset = datasets.ImageFolder(train_data_dir, dataTrans)
    val_dataset = datasets.ImageFolder(val_data_dir, dataTrans)

    image_datasets = dict(train=train_dataset, val=val_dataset)

    dataloaders = {
        x: torch.utils.data.DataLoader(
            dataset=image_datasets[x],
            batch_size=16,
            shuffle=True,
            num_workers=24
        ) for x in ['train', 'val']
    }

    dataloaders['train_size'] = len(train_dataset)
    dataloaders['val_size'] = len(val_dataset)
    dataloaders['datasets'] = image_datasets
    dataloaders['classes'] = len(train_dataset.classes)
    # data_loader_train = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
    #                                                 num_workers=os.cpu_count())
    # data_loader_val = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True,
    #                                               drop_last=True, num_workers=os.cpu_count())
    return train_dataset, val_dataset, dataloaders


def read_test_data(data_dir, batch_size):
    val_data_dir = f'{data_dir}/val'
    # test_data_dir = f'{data_dir}/test'
    # 这里可自行修改数据预处理，batch大小也可自行调整
    # 保持本地训练的数据读取和这里一致
    val_dataset = datasets.ImageFolder(val_data_dir, dataTrans)
    image_datasets = dict(val=val_dataset)
    dataloaders = {
        x: torch.utils.data.DataLoader(
            dataset=image_datasets[x],
            batch_size=16,
            shuffle=True,
            num_workers=24
        ) for x in ['val']
    }
    dataloaders['val_size'] = len(val_dataset)
    dataloaders['datasets'] = image_datasets
    dataloaders['classes'] = len(val_dataset.classes)
    # data_loader_train = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
    #                                                 num_workers=os.cpu_count())
    # data_loader_val = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True,
    #                                               drop_last=True, num_workers=os.cpu_count())
    return val_dataset, dataloaders
