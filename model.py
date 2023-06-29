from torchvision import models
import torch
import torch.nn as nn


def model(model_name:str, stage: str, dataloaders, model_path: str = ''):
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, dataloaders['classes'])
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, dataloaders['classes'])
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, dataloaders['classes'])
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, dataloaders['classes'])
    elif model_name == 'convnext_tiny':
        model = models.convnext_tiny(pretrained=True)
        num_ftrs = model.classifier[2].in_features
        model.fc = nn.Linear(num_ftrs, dataloaders['classes'])
    elif model_name == 'convnext_base':
        model = models.convnext_base(pretrained=True)
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, dataloaders['classes'])
    elif model_name == 'google_net':
        model = models.googlenet(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, dataloaders['classes'])
    elif model_name == 'convnext_large':
        model = models.convnext_large(pretrained=True)
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, dataloaders['classes'])
    if stage == 'test':
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
    return model
