# VTBench: Evaluating Visual Tokenizers for Autoregressive Image Generation
<div align="center">
  
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/huawei-lin/LLMsEasyFinetune/blob/master/LICENSE)&nbsp;[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)&nbsp;![Static Badge](https://img.shields.io/badge/VTBench-yellow?logoSize=4&label=%F0%9F%A4%97%20Dataset&labelColor=gray&link=https%3A%2F%2Fhuggingface.co%2Fdatasets%2Fhuaweilin%2FVTBench)

</div>

This repository provides the official implementation of **VTBench**, a benchmark designed to evaluate the performance of **visual tokenizers (VTs)** in the context of **autoregressive (AR) image generation**. VTBench enables fine-grained analysis across three core tasks: **image reconstruction**, **detail preservation**, and **text preservation**, isolating the tokenizer's impact from the downstream generation model.


---

## Why VTBench?

Recent AR models such as GPT-4o demonstrate impressive image generation quality, which we hypothesize is made possible by a **highly capable visual tokenizer**. However, most existing VTs significantly lag behind **continuous VAEs**, leading to:

- Poor reconstruction fidelity  
- Loss of structural and semantic detail  
- Failure to preserve symbolic information (e.g., text in multilingual images)

VTBench isolates and evaluates VT quality, independent of the downstream model, using standardized tasks and metrics.

---

## Features

- Evaluation on **three tasks**:
  1. **Image Reconstruction** (ImageNet, High-Res, Varying-Res)
  2. **Detail Preservation** (patterns, fine textures)
  3. **Text Preservation** (posters, academic abstracts, multilingual scripts)
- Supports VTs from models like **FlowMo**, **MaskBiT**, **OpenMagViT2**, **VAR**, **BSQ-ViT**, etc.
- Includes baselines from **continuous VAEs** (e.g., SD3.5L, FLUX.1) and GPT-4o.
- Metrics: PSNR, SSIM, LPIPS, FID, CER, WER

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/huawei-lin/VTBench.git
cd VTBench
```

### 2. Install dependencies
```
conda create -n vtbench python=3.10
conda activate vtbench
pip install -r requirements.txt
```


## Citation

If you find this project useful, please consider citing:
```
```
