import torch
from torch.optim import Adam
from torch.utils.data import random_split
from os.path import join

from utils import *
from run_config import *

import pretrain

assert torch.cuda.is_available(), "CUDA support is required"

if __name__=="__main__":
    torch.manual_seed(7843718)
    
    ### Pretraining ###
    print("#" * 10 + " Pretraining " + "#" * 10 + "\n")

    criterion = InPaintingLoss()

    ## Synthetic data pretraining ##

    print("Pretraining on synthetic dataset")

    synth_ds = SynthDataset(ROOT_DIR, image_size=IMAGE_SIZE)

    pretrain_ds = PretrainingDataset(synth_ds, mask_generator=MASK_GENERATOR)
    train_ds, val_ds, test_ds = random_split(pretrain_ds, [SPLIT * SPLIT, SPLIT * (1 - SPLIT), 1 - SPLIT])
    print(f"Number of training examples: {len(train_ds)}")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_dl = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_dl = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = MODEL_CLASS(PRETRAIN_NUM_OUT_CHANNELS).to(DEVICE)
    optim = Adam(model.parameters(), lr=LR)

    pretrain.train_loop(
        train_dl=train_dl,
        val_dl=val_dl,
        model=model,
        criterion=criterion,
        optim=optim,
        max_num_epochs=PRETRAIN_MAX_NUM_EPOCHS,
        patience=PATIENCE,
    )

    print("Done\n")

    pretrain_image_output(model, test_dl, os.path.join(EXAMPLE_IMAGES_DIR, SYNTH_PRETRAIN_NAME + ".jpg"), DEVICE)

    model = model.to(dtype=torch.float32)
    torch.save(model.state_dict(), os.path.join(SAVED_MODEL_DIR, SYNTH_PRETRAIN_NAME + ".pt"))

    ## Kaggle dogs and cats pretraining ##

    print("Pretraining on Kaggle cats and dogs dataset")

    train_val_ds = PretrainingDataset(
        KaggleDogsAndCats(ROOT_DIR, split="train", image_size=IMAGE_SIZE), mask_generator=MASK_GENERATOR
    )
    test_ds = PretrainingDataset(
        KaggleDogsAndCats(ROOT_DIR, split="test", image_size=IMAGE_SIZE), mask_generator=MASK_GENERATOR
    )

    train_ds, val_ds = random_split(train_val_ds, [SPLIT, 1 - SPLIT])
    print(f"Number of training examples: {len(train_ds)}")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_dl = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_dl = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = MODEL_CLASS(PRETRAIN_NUM_OUT_CHANNELS).to(DEVICE)
    optim = Adam(model.parameters(), lr=LR)

    pretrain.train_loop(
        train_dl=train_dl,
        val_dl=val_dl,
        model=model,
        criterion=criterion,
        optim=optim,
        max_num_epochs=PRETRAIN_MAX_NUM_EPOCHS,
        patience=PATIENCE,
    )

    print("Done\n")

    pretrain_image_output(model, test_dl, os.path.join(EXAMPLE_IMAGES_DIR, KAGGLE_PRETRAIN_NAME + ".jpg"), DEVICE)

    model = model.to(dtype=torch.float32)
    torch.save(model.state_dict(), os.path.join(SAVED_MODEL_DIR, KAGGLE_PRETRAIN_NAME + ".pt"))