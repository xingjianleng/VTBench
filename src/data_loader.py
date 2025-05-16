import PIL
from PIL import Image
from dataclasses import dataclass, field
from datasets import load_dataset
import torch
from .data_processing import pil_to_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, dataset_name, **kwargs):
        override_params = {}
        if dataset_name == "DIV2K":
            override_params = {
                "target_image_size": -1,
                "lock_ratio": True,
                "center_crop": False,
                "padding": False,
            }
        if dataset_name == "imagenet":
            override_params = {"center_crop": True, "padding": False}
        if dataset_name == "movie_posters":
            override_params = {"center_crop": True, "padding": False}
        if dataset_name == "high_quality_1024":
            override_params = {"target_image_size": (1024, 1024)}

        self.data_params = {**kwargs, **override_params}

    def __call__(self, instances):
        images = torch.stack(
            [
                pil_to_tensor(instance["image"], **self.data_params)
                for instance in instances
            ],
            dim=0,
        )
        idx = [instance["idx"] for instance in instances]
        return dict(image=images, idx=idx)


class ImagenetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, split_name="test", n_take=None):
        print(dataset_name, split_name)
        ds = load_dataset("huaweilin/VTBench", name=dataset_name, split=split_name if n_take is None else f"{split_name}[:{n_take}]")
        self.image_list = ds["image"]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        return dict(
            image=self.image_list[idx],
            idx=idx,
        )


def get_dataset(dataset_name, split_name, n_take):
    dataset = ImagenetDataset(dataset_name, split_name, n_take)
    return dataset
