import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import random_split

from utils import *
from run_config import *

import segmentation_train as train

assert torch.cuda.is_available(), "CUDA support is required"

if __name__=="__main__":
    torch.manual_seed(1537890)
    
    print("#" * 10 + " Image segmentation training " + "#" * 10 + "\n")

    train_val_ds = OxfordPetsDataset(ROOT_DIR, split="train", image_size=IMAGE_SIZE)

    train_sample_splits = [1.0, 0.75, 0.5, 0.25, 0.1]

    for train_split in train_sample_splits:
        train_val_ds, _ = random_split(train_val_ds, [train_split, 1 - train_split])

        test_ds = OxfordPetsDataset(ROOT_DIR, split="test", image_size=IMAGE_SIZE)
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

        train.train_loop(
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

        train.train_loop(
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

        train.train_loop(
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
