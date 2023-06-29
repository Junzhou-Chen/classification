from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
# trainTrans = transforms.Compose([
#     transforms.Resize([299, 299]),  # resize  长宽不一致
#     # transforms.CenterCrop(224),  # 在图片的中间区域进行裁剪
#     transforms.RandomRotation(45),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),  # 将PIL image转成(c,h,w)的tensor格式，并除255进行归一化处理
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 逐通道进行标准化，均值变为0，标准差变为1
# ])
#
# dataTrans = transforms.Compose([
#     transforms.Resize([299, 299]),  # resize  长宽不一致
#     # transforms.CenterCrop(224),  # 在图片的中间区域进行裁剪
#     transforms.ToTensor(),  # 将PIL image转成(c,h,w)的tensor格式，并除255进行归一化处理
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 逐通道进行标准化，均值变为0，标准差变为1
# ])

# 数据路径
data_dir = r'G:\Dataset\classification'
train_data_dir = f'{data_dir}/train'
val_data_dir = f'{data_dir}/val'
# test_data_dir = f'{data_dir}/test'

# 通用数据加载器
train_dataset = datasets.ImageFolder(train_data_dir)
val_dataset = datasets.ImageFolder(val_data_dir)
# test_dataset = datasets.ImageFolder(test_data_dir)

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
# dataloaders['test_size'] = len(test_dataset)
dataloaders['datasets'] = image_datasets
dataloaders['classes'] = len(train_dataset.classes)
