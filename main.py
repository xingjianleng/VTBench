import numpy as np
import os
import PIL
import pickle
import torch
import argparse
import json
from PIL import Image
import torch.nn as nn
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from src.data_loader import DataCollatorForSupervisedDataset, get_dataset
from src.data_processing import tensor_to_pil
from src.model_processing import get_model
from PIL import Image
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="chameleon")
parser.add_argument("--model_path", type=str, default=None)
parser.add_argument("--dataset_name", type=str, default="task3-movie-posters")
parser.add_argument("--split_name", type=str, default="test")
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--begin_id", default=0, type=int)
parser.add_argument("--n_take", default=-1, type=int)
args = parser.parse_args()

batch_size = args.batch_size
output_dir = args.output_dir

accelerator = Accelerator()

if accelerator.is_main_process and output_dir is not None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/original_images", exist_ok=True)
    os.makedirs(f"{output_dir}/reconstructed_images", exist_ok=True)
    os.makedirs(f"{output_dir}/results", exist_ok=True)

model, data_params = get_model(args.model_path, args.model_name)
dataset = get_dataset(args.dataset_name, args.split_name, None if args.n_take <= 0 else args.n_take)
data_collator = DataCollatorForSupervisedDataset(args.dataset_name, **data_params)
dataloader = DataLoader(
    dataset, batch_size=batch_size, num_workers=0, collate_fn=data_collator
)

model, dataloader = accelerator.prepare(model, dataloader)
print("Model prepared...")


def save_results(
    pixel_values, reconstructed_image, idx, output_dir, data_params
):
    if reconstructed_image is None:
        return

    ori_img = tensor_to_pil(pixel_values, **data_params)
    rec_img = tensor_to_pil(reconstructed_image, **data_params)

    ori_img.save(f"{output_dir}/original_images/{idx:08d}.png")
    rec_img.save(f"{output_dir}/reconstructed_images/{idx:08d}.png")

    result = {
        "ori_img": ori_img,
        "rec_img": rec_img,
    }

    with open(f"{output_dir}/results/{idx:08d}.pickle", "wb") as fw:
        pickle.dump(result, fw)


executor = ThreadPoolExecutor(max_workers=16)
with torch.no_grad():
    print("Begin data loading...")
    for batch in tqdm(dataloader):
        pixel_values = batch["image"]
        reconstructed_images = model(pixel_values)
        if isinstance(reconstructed_images, tuple):
            reconstructed_images = reconstructed_images[0]

        if output_dir is not None:
            idx_list = batch["idx"]
            original_images = pixel_values.detach().cpu()
            if not isinstance(reconstructed_images, list):
                reconstructed_images = reconstructed_images.detach().cpu()
            for i in range(pixel_values.shape[0]):
                executor.submit(
                    save_results,
                    original_images[i],
                    reconstructed_images[i],
                    idx_list[i],
                    output_dir,
                    data_params,
                )

executor.shutdown(wait=True)
