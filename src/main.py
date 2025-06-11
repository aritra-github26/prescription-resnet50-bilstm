"""
Provides options via the command line to perform project tasks.
* `--source`: dataset/model name (bentham, iam, rimes, saintgall, washington)
* `--transform`: transform dataset to the HDF5 file
"""
from pathlib import Path

# import torch

import numpy as np
import argparse
# import cv2
import h5py
import os
import string
import torchvision.transforms as T

from data import preproc as pp, evaluation
from data.generator import DataGenerator, Tokenizer
from data.reader import Dataset



# import torch.nn as nn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)

    parser.add_argument("--transform", action="store_true", default=False, help="Transform dataset to HDF5 file format")


    
    args = parser.parse_args()

    raw_path = os.path.join("..", "raw", args.source)
    source_path = os.path.join("..", "data", f"{args.source}.hdf5")
    output_path = os.path.join("..", "output", args.source,)
    target_path = os.path.join(output_path, "checkpoint_weights.pt")

    input_size = (1024, 128, 1)
    max_text_length = 128
    charset_base = string.printable[:95]
    tokenizer = Tokenizer(chars=charset_base, max_text_length=max_text_length)            

    if args.transform:
        print(f"{args.source} dataset will be transformed into HDF5 format...")

        ds = Dataset(source=raw_path, name=args.source)
        ds.read_partitions()

        print("Partitions will be preprocessed for HDF5 storage...")

        ds.preprocess_partitions(input_size=input_size)

        print("Partitions will be saved to HDF5 file...")

        os.makedirs(os.path.dirname(source_path), exist_ok=True)

        for i in ds.partitions:
            with h5py.File(source_path, "a") as hf:
                hf.create_dataset(f"{i}/dt", data=ds.dataset[i]['dt'], compression="gzip", compression_opts=9)
                hf.create_dataset(f"{i}/gt", data=ds.dataset[i]['gt'], compression="gzip", compression_opts=9)
                print(f"[OK] {i} partition.")

        print(f"Transformation to HDF5 format finished.")

    
