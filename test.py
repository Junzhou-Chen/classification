import argparse
import logging
from utils.utils import *
from sklearn.metrics import accuracy_score
from model import model


def get_args():
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--load', '-f', type=str, default='./pth/resnet50_best.pth', help='Load model from a .pth file')
    parser.add_argument('--model', '-m', type=str, default='resnet50',
                        help='Choose from resnet50, resnet101, resnet152, alexnet, convnext_tiny, connect_base, '
                             'google_net, convnext_large')
    parser.add_argument('--test-path', '-p', dest='test_path', type=str, default=r'G:\Dataset\Herlev-cls',
                        help='Dataset path')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=5, help='Number of classes')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    dataset_val, dataset_loader = read_test_data(args.data_path, args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', )

    model_ftrs = model(args.model, stage='test', dataloaders=dataset_loader, model_path=args.load)
    model_ftrs.to(device=device)
    model_ftrs.eval()
    logging.info(f'Using model: {args.model}')

    val_loader = DataLoader(dataset=dataset_val, batch_size=args.batch_size, drop_last=True, shuffle=False)
    num_val_batches = len(val_loader)
    f1_score = 0
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=args.amp):
        for idx, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            # predict the mask
            pred = model_ftrs(inputs)
            _, predicted = torch.max(pred.data, 1)
            acc = accuracy_score(labels.cpu(), predicted.cpu())
            f1_score += acc

    test_score = f1_score / max(num_val_batches, 1)

    logging.info(f'F1 score: {test_score}')

    print('F1 score: ', test_score)




