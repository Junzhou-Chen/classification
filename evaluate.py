import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
# from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    f1_score = 0
    dice_score1 = 0
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            # predict the mask
            pred = net(inputs)
            _, predicted = torch.max(pred.data, 1)
            precision, recall, f1, support = precision_recall_fscore_support(labels, predicted, average='weighted')
            f1_score += f1
            # print(labels)
    net.train()
    return f1_score / max(num_val_batches, 1)
