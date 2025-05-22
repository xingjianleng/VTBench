import os
import argparse
from PIL import Image
from tqdm import tqdm
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from ocr import OCR
from character_error_rate import CharacterErrorRate
from word_error_rate import WordErrorRate
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
    FrechetInceptionDistance,
)


class ImageFolderPairDataset(Dataset):
    def __init__(self, dir1, dir2, transform=None):
        self.dir1 = dir1
        self.dir2 = dir2
        self.filenames = sorted(os.listdir(dir1))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        img1 = Image.open(os.path.join(self.dir1, name)).convert("RGB")
        img2 = Image.open(os.path.join(self.dir2, name)).convert("RGB")
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()]
    )

    dataset = ImageFolderPairDataset(
        args.original_dir, args.reconstructed_dir, transform
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    if "cer" in args.metrics or "wer" in args.metrics:
        ocr = OCR(device)

    # Metrics init
    metrics = {}

    if "psnr" in args.metrics:
        metrics["psnr"] = PeakSignalNoiseRatio().to(device)
    if "ssim" in args.metrics:
        metrics["ssim"] = StructuralSimilarityIndexMeasure().to(device)
    if "lpips" in args.metrics:
        metrics["lpips"] = LearnedPerceptualImagePatchSimilarity().to(device)
    if "fid" in args.metrics:
        metrics["fid"] = FrechetInceptionDistance().to(device)
    if "cer" in args.metrics:
        metrics["cer"] = CharacterErrorRate(ocr)
    if "wer" in args.metrics:
        metrics["wer"] = WordErrorRate(ocr)

    for batch in tqdm(loader, desc="Evaluating"):
        # img1, img1_path, img2, img2_path = [b.to(device) for b in batch]
        img1, img2 = [b.to(device) for b in batch]

        if "psnr" in metrics:
            metrics["psnr"].update(img2, img1)
        if "ssim" in metrics:
            metrics["ssim"].update(img2, img1)
        if "lpips" in metrics:
            metrics["lpips"].update(img2, img1)
        if "cer" in metrics:
            metrics["cer"].update(img2, img1)
        if "wer" in metrics:
            metrics["wer"].update(img2, img1)
        if "fid" in metrics:
            img1_uint8 = (img1 * 255).clamp(0, 255).to(torch.uint8)
            img2_uint8 = (img2 * 255).clamp(0, 255).to(torch.uint8)
            metrics["fid"].update(img1_uint8, real=True)
            metrics["fid"].update(img2_uint8, real=False)

    print("\nResults:")
    for name, metric in metrics.items():
        print(f"{name.upper()}", end="\t")
    print()
    for name, metric in metrics.items():
        result = metric.compute().item()
        print(f"{result:.4f}", end="\t")
    print()

    metrics_df = pd.DataFrame({k: [v.compute().item()] for k, v in metrics.items()})
    metrics_df.to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original_dir", type=str, required=True, help="Path to original images"
    )
    parser.add_argument(
        "--reconstructed_dir",
        type=str,
        required=True,
        help="Path to reconstructed images",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["psnr", "ssim", "lpips", "fid"],
        help="Metrics to compute: psnr, ssim, lpips, fid",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for processing"
    )
    parser.add_argument("--image_size", type=int, default=256, help="Image resize size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for DataLoader"
    )
    args = parser.parse_args()

    evaluate(args)
