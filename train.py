import logging
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.evaluate import evaluate
import wandb


def train_model(
        model,
        device,
        train_set,
        val_set,
        args,
        epochs: int = 20,
        batch_size: int = 128,
        learning_rate: float = 1e-5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    n_train = len(train_set)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, drop_last=True, shuffle=False)

    # (Initialize logging)
    experiment = wandb.init(project=args.model, resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, amp=amp)
    )

    logging.basicConfig(
        filename=f'./logs/{args.model}.log',
        filemode='w',
        level=logging.INFO,
        format='[%(levelname)s] %(asctime)s [%(filename)s:%(lineno)d]: %(message)s'
    )

    # 4. Set up the optimi   zer, the loss, the learning rate scheduler and the loss scaling for AMP
    # params = list(model.parameters())
    # optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)

    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0
    best_score = -1
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        logging.info(f'Epoch {epochs}/{epoch}')  # 输出到日志
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for data in train_loader:
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
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
                        logging.info(f'Val Acc: {val_score:.4f}')

                        scheduler.step(val_score)
                        logging.info('Accuracy score: {}'.format(val_score))
                        state_dict = model.state_dict()
                        torch.save(state_dict, str(f'./pth/{args.model}.pth'))

                        logging.info(f'Train Loss: {epoch_loss / global_step:.4f}')
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Accuracy': val_score,
                            'step': global_step,
                            'epoch': epoch,
                            # **histograms
                        })
                        if val_score > best_score:
                            best_score = val_score
                            torch.save(state_dict, str(f'./pth/{args.model}_best.pth'))
                            logging.info(f'Best saved! score: {best_score}')




