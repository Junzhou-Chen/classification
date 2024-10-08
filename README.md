# classification



## Environment

1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch 1.13 or later](https://pytorch.org/get-started/locally/)

3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

## Training

**Note : Use Python 3.6 or newer**

```conosle
> python main.py -h
usage: main.py [-h] [--epochs E] [--batch-size B] [--learning-rate LR] [--load LOAD] [--model MODEL]
               [--data-path DATA_PATH] [--amp] [--bilinear] [--classes CLASSES]

Train the HARUNet on images and target masks

options:
  -h, --help            show this help message and exit
  --epochs E, -e E      Number of epochs
  --batch-size B, -b B  Batch size
  --learning-rate LR, -l LR
                        Learning rate
  --load LOAD, -f LOAD  Load model from a .pth file
  --model MODEL, -m MODEL
                        Choose from resnet50, resnet101, resnet152, alexnet, convnext_tiny, connect_base, google_net,
                        convnext_large
  --data-path DATA_PATH, -p DATA_PATH
                        Dataset path
  --amp                 Use mixed precision
  --bilinear            Use bilinear upsampling
  --classes CLASSES, -c CLASSES
                        Number of classes
```





Training:

```bash
python train.py --epochs 50 --batch-size 2 --learning-rate 1e-5 --scale 0.5 --valodation 10 --amp 
```

## Prediction

After training your model and saving it to `model.pth`, you can easily test the output masks on your images via the CLI.

Predict all images in the folder and save:

```bash
python test.py -test-path input_folder 
```

## Weights & Biases

The training progress can be visualized in real-time using [Weights & Biases](https://wandb.ai/).  Loss curves, validation curves, weights and gradient histograms, as well as predicted masks are logged to the platform.

When launching a training, a link will be printed in the console. Click on it to go to your dashboard. If you have an existing W&B account, you can link it
 by setting the `WANDB_API_KEY` environment variable. If not, it will create an anonymous run which is automatically deleted after 7 days.
