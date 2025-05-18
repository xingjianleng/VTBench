import os
from omegaconf import OmegaConf
import torch
import tempfile
from safetensors.torch import load_file
import requests
import yaml

def get_ckpt(path, key="state_dict"):
    is_url = path.startswith("http://") or path.startswith("https://")
    suffix = os.path.splitext(path)[-1]

    if is_url:
        print(f"Loading checkpoint from URL: {path}")
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp_file:
            response = requests.get(path)
            response.raise_for_status()
            tmp_file.write(response.content)
            tmp_file.flush()
            ckpt_path = tmp_file.name

            if suffix == ".safetensors":
                checkpoint = load_file(ckpt_path)
            else:
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    else:
        print(f"Loading checkpoint from local path: {path}")
        if suffix == ".safetensors":
            checkpoint = load_file(path)
        else:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if key is not None and key in checkpoint:
        checkpoint = checkpoint[key]

    return checkpoint


def get_yaml_config(path):
    if path.startswith("http://") or path.startswith("https://"):
        response = requests.get(path)
        response.raise_for_status()
        config = OmegaConf.create(response.text)
    else:
        with open(path, 'r') as f:
            config = OmegaConf.load(f)
    return config
