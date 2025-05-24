import albumentations as A
import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import sigmoid
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryJaccardIndex
import time
import os
import wandb
import numpy as np
import random

from core.paths import train_dir, val_dir, get_model_path
from core.dataset import TeethDataset
from core.model import UNet


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Set random seed {seed}.")


def make_dataloader(base_dir, is_train, batch_size, transformations):
    dataset = TeethDataset(base_dir=base_dir,
                           transformations=transformations)
    # print(f"{len(dataset)} images in dataset")
    dataloader = DataLoader(dataset, shuffle=is_train, batch_size=batch_size,
                            pin_memory=True if device == "cuda" else False,
                            num_workers=os.cpu_count())
    # print(f"{len(dataloader)} batches of size {batch_size} in dataloader")
    return dataloader


def get_optimizer(model_params, config):
    if config['optimizer'] == 'adam':
        return torch.optim.Adam(model_params, lr=config['learning_rate'])
    elif config['optimizer'] == 'adamw':
        return torch.optim.AdamW(model_params,lr=config['learning_rate'], weight_decay=config['weight_decay'])
    else:
        raise ValueError(f"Optimizer {config['optimizer']} is not supported. Please use 'adam' or 'adamw'.")


def setup_model(config):
    unet = UNet().to(device)
    loss_function = config['loss_func']
    optimizer = get_optimizer(unet.parameters(), config)
    scheduler = OneCycleLR(optimizer,
                           config['learning_rate'],
                           total_steps=config['num_epochs'],
                           pct_start=config['warmup_pct']) if config['use_scheduler'] else None
    return unet, loss_function, optimizer, scheduler


def validate_model(model, dataloader, loss_function, epoch, config):
    total_loss = 0.0
    iou = BinaryJaccardIndex().to(device)
    model.eval()
    with torch.no_grad():  # no need for backpropagation
        for batch_id, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)
            predictions = model(images)
            loss = loss_function(predictions, masks)
            total_loss += loss.item()

            probs = sigmoid(predictions)
            predictions_binary = (probs > 0.1).float()
            iou.update(predictions_binary, masks)

            if config['log_images'] and epoch % config['log_images_steps'] == 0:
                wandb.log({"img": [wandb.Image(predictions[0].cpu().numpy(), caption="idx 0"),
                                   wandb.Image(predictions[3].cpu().numpy(), caption="idx 3"),
                                   wandb.Image(predictions[6].cpu().numpy(), caption="idx 6")]
                           }, step=epoch)
    total_loss /= len(dataloader)
    iou_acc = iou.compute()
    wandb.log({"val_loss": total_loss, "iou": iou_acc}, step=epoch)
    iou.reset()
    return total_loss, iou_acc


def train_for_one_epoch(epoch,
                        unet,
                        train_dl,
                        optimizer,
                        loss_function,
                        scheduler):
    epoch_loss = 0.0
    unet.train()  # set unet in training mode
    for images, masks in train_dl:  # for each batch
        (images, masks) = (images.to(device),
                           masks.to(device))
        optimizer.zero_grad()  # clear gradients
        predictions = unet(images)
        loss = loss_function(predictions, masks)
        loss.backward()  # backpropagation, gradients wrt all variables
        optimizer.step()  # update the model's parameters
        epoch_loss += loss.item()
    if scheduler is not None:
        scheduler.step()
        #[lr] = scheduler.get_last_lr()
        #wandb.log({"lr": lr}, step=epoch, commit=False)
    avg_loss = epoch_loss / len(train_dl)
    wandb.log({"train_loss": avg_loss}, step=epoch)
    return avg_loss


def train_model(config):
    transformations = A.Compose([A.HorizontalFlip(p=.5),
                                 A.Rotate(45, p=.4),
                                 A.Perspective(p=.4),
                                 A.RandomResizedCrop(height=config['input_size'],
                                                     width=config['input_size'],
                                                     scale=(.4,1.),
                                                     ratio=(1,1),
                                                     p=1.),
                                 A.RandomBrightnessContrast(p=.5),
                                 A.ColorJitter(p=.5),
                                 A.PixelDropout(dropout_prob=.001, p=.5)
                                 ])
    val_transformations = A.Compose([A.Resize(height=config['input_size'],
                                              width=config['input_size'],
                                              p=1)])

    train_dl = make_dataloader(base_dir=train_dir,
                               is_train=True,
                               batch_size=config['batch_size'],
                               transformations=transformations)
    val_dl = make_dataloader(base_dir=val_dir,
                             is_train=False,
                             batch_size=config['batch_size'],
                             transformations=val_transformations)

    unet, loss_function, optimizer, scheduler = setup_model(config)

    wandb.init(project="braces_project",
               name=config['run_name'],
               notes=config['logging_note'],
               config=config
    )

    # training loop
    print(f"Start training the network for {config['num_epochs']} epochs", "with" if config['early_stopping'] else "without", "early stopping")

    trigger_times = 0
    previous_loss = 100.0
    best_loss = 100.0
    best_iou = 0.0
    trained_epochs = 0

    start_time = time.time()
    for epoch in range(config['num_epochs']):
        avg_loss = train_for_one_epoch(epoch, unet, train_dl, optimizer,
                                       loss_function, scheduler)
        val_loss, iou = validate_model(unet, val_dl, loss_function, epoch,
                                  config)
        print(
            f"[EP {epoch}] Avg train loss: {avg_loss:.4f}, val loss: {val_loss:.4f}")

        # early stopping
        if config['early_stopping']:
            if val_loss > previous_loss:
                trigger_times += 1
                if trigger_times >= config['patience']:
                    trained_epochs = epoch
                    print(f"Stopped training early at epoch {epoch}")
                    break
            else:
                trigger_times = 0
            previous_loss = val_loss

        if epoch > 100:
            if iou > best_iou and val_loss < best_loss:
                state = {
                    'state_dict': unet.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epochs': epoch
                }
                torch.save(state, get_model_path(str(config['RUN'])+"-ep"+str(epoch)))
                best_iou = iou
                best_loss = val_loss

    if not config['early_stopping']:
        trained_epochs = config['num_epochs']
    end_time = time.time()
    print(f"Time taken to train the model: {end_time - start_time:.2f}s")
    wandb.finish()

    state = {
        'state_dict': unet.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epochs': trained_epochs
    }
    torch.save(state, get_model_path(config['run_name']))


if __name__ == '__main__':
    #set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] used {device} device.")

    RUN = 30

    base_config = {
        'RUN': RUN,
        'loss_func': BCEWithLogitsLoss(),
        'learning_rate': 0.01,
        'num_epochs': 120,
        'batch_size': 16,
        'input_size': 256,
        'optimizer': 'adamw',
        'weight_decay': 0.001,
        'use_scheduler': True,
        'warmup_pct': 0.2,
        'early_stopping': False,
        'patience': 4,
        'run_name': f"ex_{RUN}",
        'logging_note': "",
        'log_images': False,
        'log_images_steps': 10
    }
    # experiments = []
    # learning_rates = [.01, .02, .03]
    # optimizers = ['adam', 'adamw']
    # warmups = [.1, .2, .3]
    # weight_decay = [.1, .01, .001]
    #
    # for lr in learning_rates:
    #     for wd in weight_decay:
    #         config = base_config.copy()
    #         config.update({
    #             'learning_rate': lr,
    #             'weight_decay': wd,
    #             'RUN': str(RUN)+'-'+str(len(experiments)),
    #             'run_name': f"ex_{RUN}-{len(experiments)}-lr-wd"
    #         })
    #         experiments.append(config)

    for i in range(6):
        base_config.update({'run_name': f"ex{RUN}-i"})
        train_model(base_config)
