import argparse
import logging
from utils.utils import *
from train import train_model
from model import model


def get_args():
    parser = argparse.ArgumentParser(description='Train the HARUNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-6,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--model', '-m', type=str, default='resnet50',
                        help='Choose from resnet50, resnet101, resnet152, alexnet, convnext_tiny, connect_base, '
                             'google_net, convnext_large')
    parser.add_argument('--data-path', '-p', dest='data_path', type=str, default=r'G:\Dataset\Herlev-cls',
                        help='Dataset path')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=5, help='Number of classes')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    dataset_train, dataset_val, dataset_loader = read_data(args.data_path, args.batch_size)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model_ftrs = model(args.model, stage='train', dataloaders=dataset_loader)
    model_ftrs.to(device=device)
    logging.info(f'Using model: {args.model}')
    try:
        train_model(
            model=model_ftrs,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            train_set=dataset_train,
            val_set=dataset_val,
            device=device,
            amp=args.amp,
            args=args,
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()

        train_model(
            model=model_ftrs,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            train_set=dataset_train,
            val_set=dataset_val,
            device=device,
            amp=args.amp,
            args=args,
        )
