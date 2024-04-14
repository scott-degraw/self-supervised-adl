""" 
Script to generate test intersection over union test dataset statistics for each run and
training size sample. Saves to csv file in SAVED_MODEL_DIR/TEST_IOUS_FNAME.
"""

from glob import glob
from os import path
import csv
import re

import torch

from utils import *
from run_config import *

TEST_IOUS_FNAME = "test_ious.csv"

if __name__ == "__main__":
    test_ds = OxfordPetsDataset(root=ROOT_DIR, split="test", image_size=IMAGE_SIZE)
    test_dl = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    kaggle_models = glob(r"kaggle_seg_size*.pt", root_dir=SAVED_MODEL_DIR)
    synth_models = glob(r"synth_seg_size*.pt", root_dir=SAVED_MODEL_DIR)
    no_pretrain_models = glob(r"no_pretrain_size*.pt", root_dir=SAVED_MODEL_DIR)

    model_types = ("kaggle_seg", "synth_seg", "no_pretrain")
    model_type_fnames = (kaggle_models, synth_models, no_pretrain_models)

    columns = ["run", "train_size", "model_type", "test_iou"]

    with open(path.join(SAVED_MODEL_DIR, TEST_IOUS_FNAME), "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for model_name, model_fnames in zip(model_types, model_type_fnames):
            for model_fname in model_fnames:
                # Use regex on filename to find the train size and run number
                regex = model_name + r"_size_(\d+)_run_(\d).pt"
                match = re.search(regex, model_fname)
                train_size = match.group(1)
                run = match.group(2)

                model = MODEL_CLASS(num_out_channels=1)
                model_path = path.join(SAVED_MODEL_DIR, model_fname)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model = model.to(DEVICE)
                
                test_iou = model_iou(model, test_dl, DEVICE)
                print(f"{model_name}, train_size: {train_size}, run: {run}, test IOU: {test_iou:.4g}")

                writer.writerow(
                    {
                        "run": run,
                        "train_size": train_size,
                        "model_type": model_name,
                        "test_iou": test_iou,
                    }
                )
