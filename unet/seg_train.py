from os.path import join

import torch
import torch.nn as nn

from torch.optim import Adam 
from torch.cuda.amp import autocast

from torch.utils.data import random_split

from utils import * 
from run_config import *

'''
Training/Testing Loops
''' 
def epoch_step(train_dl:torch.utils.data.DataLoader, model:nn.Module, criterion:nn.Module, optimizer:torch.optim.Optimizer) -> float:
    '''
    Do one epoch training Step
    Inputs:
        - train_dl (data.DataLoader): training dataloader
        - model (nn.Module): model to train
        - criterion (nn.Module): loss function
        - optimizer (optim.Optimizer): optimizer
    Returns:
        - total_loss: total loss for the epoch
    '''
    total_loss = 0.
    model.train()
    for (inputs, targets) in train_dl:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        # Forward pass
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += inputs.shape[0] * loss.item()

        # Backward pass and Optimize
        SCALER.scale(loss).backward()
        SCALER.step(optimizer)
        SCALER.update()

    return total_loss / len(train_dl.dataset)


def test_step(test_dl:torch.utils.data.DataLoader, model:nn.Module, criterion:nn.Module) -> float:
    '''
    Test using the validation/test set
    Inputs:
        - test_dl (data.DataLoader): test dataloader
        - model (nn.Module): model to test
        - criterion (nn.Module): loss function
    Returns:
        Loss for the test set
    '''
    total_loss = 0.
    model.eval()
    with torch.no_grad():
        for (inputs, targets) in test_dl:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            # Forward pass
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += targets.shape[0] * loss.item()

    return total_loss / len(test_dl.dataset)

def train_loop(
    train_dl: DataLoader,
    val_dl: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optim: torch.optim.Optimizer,
    max_num_epochs: int = 10,
    patience: int = 5,
):
    val_scores = []
    best_val_score = -torch.inf
    no_improvement_counter = 0

    model_state_dicts = []

    for epoch in range(max_num_epochs):
        # Train
        epoch_loss = epoch_step(train_dl, model, criterion, optim)
        # Test
        val_score = model_iou(model, val_dl, DEVICE)
        print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4g} Val IOU: {val_score:.4g}")

        # Early stopping
        val_scores.append(val_score)

        cpu_state_dict = {key: tensor.cpu() for key, tensor in model.state_dict().items()}
        model_state_dicts.append(cpu_state_dict)

        if val_score > best_val_score:
            no_improvement_counter = 0
            best_val_score = val_score
        else:
            no_improvement_counter += 1

        if no_improvement_counter == patience:
            break

    best_epoch = torch.tensor(val_scores).argmax()
    best_val_score = val_scores[best_epoch]
    model.load_state_dict(model_state_dicts[best_epoch])

if __name__=="__main__":
    torch.manual_seed(1537890)
    
    print("#" * 10 + " Image segmentation training " + "#" * 10 + "\n")

    full_train_val_ds = OxfordPetsDataset(ROOT_DIR, split="train", image_size=IMAGE_SIZE)
    test_ds = OxfordPetsDataset(ROOT_DIR, split="test", image_size=IMAGE_SIZE)

    train_sample_splits = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05]

    for train_split in train_sample_splits:
        if train_split != 1.0:
            train_val_ds, _ = random_split(full_train_val_ds, [train_split, 1 - train_split])
        else:
            train_val_ds = full_train_val_ds

        train_ds, val_ds = random_split(train_val_ds, [SPLIT, 1 - SPLIT])
        print(f"Number of training examples: {len(train_ds)}")

        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_dl = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        test_dl = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        criterion = nn.BCEWithLogitsLoss()

        ## No pretraining ##

        print("No pretraining")

        model = MODEL_CLASS(SEG_NUM_OUT_CHANNELS).to(DEVICE)
        optim = Adam(model.parameters(), lr=LR)

        train_loop(
            train_dl=train_dl,
            val_dl=val_dl,
            model=model,
            criterion=criterion,
            optim=optim,
            max_num_epochs=TRAIN_MAX_NUM_EPOCHS,
            patience=PATIENCE,
        )

        test_score = model_iou(model, test_dl, DEVICE)
        print(f"Test IOU: {test_score:.4g}")

        segmentation_image_output(model, test_dl, os.path.join(EXAMPLE_IMAGES_DIR, NO_PRETRAIN_SEG_NAME + ".jpg"), DEVICE)

        model = model.to(dtype=torch.float32)
        torch.save(model.state_dict(), os.path.join(SAVED_MODEL_DIR, NO_PRETRAIN_SEG_NAME + ".pt"))

        print("Done\n")

        ## Kaggle pretrain segmentation ##

        print("Image segmentation train with Kaggle dogs and cats pretrain")
        model = MODEL_CLASS(PRETRAIN_NUM_OUT_CHANNELS)
        model.load_state_dict(torch.load(os.path.join(SAVED_MODEL_DIR, KAGGLE_PRETRAIN_NAME + ".pt")))
        model.new_head(SEG_NUM_OUT_CHANNELS)
        model = model.to(DEVICE)
        optim = Adam(model.parameters(), lr=LR)

        train_loop(
            train_dl=train_dl,
            val_dl=val_dl,
            model=model,
            criterion=criterion,
            optim=optim,
            max_num_epochs=TRAIN_MAX_NUM_EPOCHS,
            patience=PATIENCE,
        )

        test_score = model_iou(model, test_dl, DEVICE)
        print(f"Test IOU: {test_score:.4g}")

        segmentation_image_output(model, test_dl, os.path.join(EXAMPLE_IMAGES_DIR, KAGGLE_SEG_NAME + ".jpg"), DEVICE)

        model = model.to(dtype=torch.float32)
        torch.save(model.state_dict(), os.path.join(SAVED_MODEL_DIR, KAGGLE_SEG_NAME + ".pt"))

        print("Done\n")

        ## Synthetic pretrain segmentation ##

        print("Image segmentation train with synthetic data pretrain")
        model = MODEL_CLASS(PRETRAIN_NUM_OUT_CHANNELS)
        model.load_state_dict(torch.load(os.path.join(SAVED_MODEL_DIR, SYNTH_PRETRAIN_NAME + ".pt")))
        model.new_head(SEG_NUM_OUT_CHANNELS)
        model = model.to(DEVICE)
        optim = Adam(model.parameters(), lr=LR)

        train_loop(
            train_dl=train_dl,
            val_dl=val_dl,
            model=model,
            criterion=criterion,
            optim=optim,
            max_num_epochs=TRAIN_MAX_NUM_EPOCHS,
            patience=PATIENCE,
        )

        test_score = model_iou(model, test_dl, DEVICE)
        print(f"Test IOU: {test_score:.4g}")

        segmentation_image_output(model, test_dl, os.path.join(EXAMPLE_IMAGES_DIR, SYNTH_SEG_NAME + ".jpg"), DEVICE)

        model = model.to(dtype=torch.float32)
        torch.save(model.state_dict(), os.path.join(SAVED_MODEL_DIR, SYNTH_SEG_NAME + ".pt"))

        print("Done\n")
