import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from evaluate import evaluate
import wandb


def train_model(
        model,
        device,
        train_set,
        val_set,
        epochs: int = 20,
        batch_size: int = 128,
        learning_rate: float = 1e-5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    n_val = len(val_set)
    n_train = len(train_set)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)

    # loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    # train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    # val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='ResNet', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, amp=amp)
    )

    logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {n_train}
            Validation size: {n_val}
            Device:          {device.type}
            Mixed Precision: {amp}
        ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0
    best_score = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for idx, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.cuda()
                labels = labels.cuda()
                # print(1)
                # outputs = model(inputs.view(inputs.size(0), -1))
                pred = model(inputs)

                loss = criterion(pred, labels)
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(inputs.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        # print(1)
                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)
                        logging.info('Accuracy score: {}'.format(val_score))
                        state_dict = model.state_dict()
                        torch.save(state_dict, str('model.pth'))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Accuracy': val_score,
                                'step': global_step,
                                'epoch': epoch,
                                # **histograms
                            })
                        except:
                            pass
                        if val_score > best_score:
                            best_score = val_score
                            torch.save(state_dict, str('best_model.pth'))
                            logging.info(f'Best saved! score: {best_score}')

