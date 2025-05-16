import os
from omegaconf import OmegaConf
import torch
import tempfile
from safetensors.torch import load_file
import requests
import yaml


def load_ckpt_from_internet(path, key="state_dict"):
    print(f"Loading checkpoint from URL: {path}")
    suffix = os.path.splitext(path)[-1]

    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp_file:
        response = requests.get(path)
        response.raise_for_status()
        tmp_file.write(response.content)
        tmp_file.flush()
        ckpt_path = tmp_file.name

        # Choose the loader based on file extension
        if suffix == ".safetensors":
            checkpoint = load_file(ckpt_path)
        else:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if key is not None and key in checkpoint:
            checkpoint = checkpoint[key]

    return checkpoint


def get_yaml_from_internet(path):
    response = requests.get(path)
    response.raise_for_status()
    config = OmegaConf.create(response.text)
    return config
