import os
import pandas as pd

root_dir = "./"

model_name_mapping = {
    "flowmo_lo": "FlowMo Lo",
    "flowmo_hi": "FlowMo Hi",
    "gpt4o": "GPT-4o",
    "janus_pro_1b": "Janus Pro 1B/7B",
    "llamagen-ds8": "LlamaGen ds8",
    "llamagen-ds16": "LlamaGen ds16",
    "llamagen-ds16-t2i": "LlamaGen ds16 T2I",
    "maskbit_16bit": "MaskBiT 16bit",
    "maskbit_18bit": "MaskBiT 18bit",
    "open_magvit2": "OpenMagViT",
    "titok_b64": "Titok-b64",
    "titok_bl64": "Titok-bl64",
    "titok_s128": "Titok-s128",
    "titok_bl128": "Titok-bl128",
    "titok_l32": "Titok-l32",
    "titok_sl256": "Titok-sl256",
    "var_256": "VAR-256",
    "var_512": "VAR-512",
    "SD3.5L": "SD3.5L",
    "FLUX.1-dev": "FLUX.1-dev",
    "infinity_d32": "Infinity-d32",
    "infinity_d64": "Infinity-d64",
    "chameleon": "Chameleon",
    "bsqvit": "BSQ-VIT",
}

output_order = [
    "FlowMo Lo",
    "FlowMo Hi",
    "MaskBiT 16bit",
    "MaskBiT 18bit",
    "Titok-l32",
    "Titok-b64",
    "Titok-s128",
    "Titok-bl64",
    "Titok-bl128",
    "Titok-sl256",
    "OpenMagViT",
    "LlamaGen ds8",
    "BSQ-VIT",
    "VAR-256",
    "Janus Pro 1B/7B",
    "Chameleon",
    "LlamaGen ds16",
    "LlamaGen ds16 T2I",
    "VAR-512",
    "Infinity-d32",
    "Infinity-d64",
    "SD3.5L",
    "FLUX.1-dev",
    "GPT-4o",
]

for dataset_name in os.listdir(root_dir):
    dataset_path = os.path.join(root_dir, dataset_name)
    if not os.path.isdir(dataset_path):
        continue

    results = {}

    for model_dir in os.listdir(dataset_path):
        model_path = os.path.join(dataset_path, model_dir)
        result_file = os.path.join(model_path, "result.txt")

        if os.path.isfile(result_file):
            with open(result_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if len(lines) >= 2:
                metrics_line = lines[-2].strip()
                values_line = lines[-1].strip()

                metrics = metrics_line.split()
                values = values_line.split()

                mapped_name = model_name_mapping.get(model_dir, model_dir)
                results[mapped_name] = values

    if results:
        header = "\t".join(metrics)
        print(f"{dataset_name}\t{header}")
        for model_name in output_order:
            if model_name in results:
                values = results[model_name]
                print(f"{model_name}\t" + "\t".join(values))
            else:
                print(f"{model_name}\t" + "no result")
        print()
