import torch
from sklearn.metrics import accuracy_score
# from utils.dice_score import multiclass_dice_coeff, dice_coeff
import warnings

warnings.filterwarnings("ignore")


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    f1_score = 0
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            # predict the mask
            pred = net(inputs)
            _, predicted = torch.max(pred.data, 1)
            acc = accuracy_score(labels.cpu(), predicted.cpu())
            f1_score += acc

    net.train()
    return f1_score / max(num_val_batches, 1)
