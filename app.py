import os
import spaces
import subprocess
import sys

# REQUIREMENTS_FILE = "requirements.txt"
# if os.path.exists(REQUIREMENTS_FILE):
#     try:
#         print("Installing dependencies from requirements.txt...")
#         subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE])
#         print("Dependencies installed successfully.")
#     except subprocess.CalledProcessError as e:
#         print(f"Failed to install dependencies: {e}")
# else:
#     print("requirements.txt not found.")

import gradio as gr
from src.data_processing import pil_to_tensor, tensor_to_pil
from PIL import Image
from src.model_processing import get_model
from huggingface_hub import snapshot_download
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

MODEL_DIR = "./VTBench_models"
if not os.path.exists(MODEL_DIR):
    print("Downloading VTBench_models from Hugging Face...")
    snapshot_download(
        repo_id="huaweilin/VTBench_models",
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False
    )
    print("Download complete.")

example_image_paths = [f"assets/app_examples/{i}.png" for i in range(0, 5)]

model_name_mapping = {
    "SD3.5L": "SD3.5L",
    "chameleon": "Chameleon",
    # "flowmo_lo": "FlowMo Lo",
    # "flowmo_hi": "FlowMo Hi",
    # "gpt4o": "GPT-4o",
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
    "FLUX.1-dev": "FLUX.1-dev",
    "infinity_d32": "Infinity-d32",
    "infinity_d64": "Infinity-d64",
    "bsqvit": "BSQ-VIT",
}

def load_model(model_name):
    model, data_params = get_model(MODEL_DIR, model_name)
    model = model.to(device)
    model.eval()
    return model, data_params

model_dict = {
    model_name: load_model(model_name)
    for model_name in model_name_mapping
}

placeholder_image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))

@spaces.GPU
def process_selected_models(uploaded_image, selected_models):
    selected_results = []
    placeholder_results = []

    for model_name in model_name_mapping:
        label = model_name_mapping[model_name]

        if uploaded_image is None:
            result = gr.update(value=placeholder_image, label=f"{label} (No input)")
        elif model_name in selected_models:
            try:
                model, data_params = model_dict[model_name]
                pixel_values = pil_to_tensor(uploaded_image, **data_params).unsqueeze(0).to(device)
                output = model(pixel_values)[0]
                reconstructed_image = tensor_to_pil(output[0].cpu(), **data_params)
                result = gr.update(value=reconstructed_image, label=label)
            except Exception as e:
                print(f"Error in model {model_name}: {e}")
                result = gr.update(value=placeholder_image, label=f"{label} (Error)")
            selected_results.append(result)
        else:
            result = gr.update(value=placeholder_image, label=f"{label} (Not selected)")
            placeholder_results.append(result)

    return selected_results + placeholder_results

with gr.Blocks() as demo:
    gr.Markdown("## VTBench")

    gr.Markdown("---")

    image_input = gr.Image(
        type="pil",
        label="Upload an image",
        width=512,
        height=512,
    )

    gr.Markdown("### Click on an example image to use it as input:")
    example_rows = [example_image_paths[i:i+5] for i in range(0, len(example_image_paths), 5)]
    for row in example_rows:
        with gr.Row():
            for path in row:
                ex_img = gr.Image(
                    value=path,
                    show_label=False,
                    interactive=True,
                    width=256,
                    height=256,
                )

                def make_loader(p=path):
                    def load_img():
                        return Image.open(p)
                    return load_img
        
                ex_img.select(fn=make_loader(), outputs=image_input)

    gr.Markdown("---")

    gr.Markdown("⚠️ **The more models you select, the longer the processing time will be.**")
    model_selector = gr.CheckboxGroup(
        choices=list(model_name_mapping.keys()),
        label="Select models to run",
        value=["SD3.5L", "chameleon", "janus_pro_1b"],
        interactive=True,
    )
    run_button = gr.Button("Start Processing")

    image_outputs = []
    model_names_ordered = list(model_name_mapping.keys())
    n_columns = 5
    output_rows = [model_names_ordered[i:i+n_columns] for i in range(0, len(model_names_ordered), n_columns)]

    with gr.Column():
        for row in output_rows:
            with gr.Row():
                for model_name in row:
                    display_name = model_name_mapping[model_name]
                    out_img = gr.Image(
                        label=f"{display_name} (Not run)",
                        value=placeholder_image,
                        width=512,
                        height=512,
                    )
                    image_outputs.append(out_img)

    run_button.click(
        fn=process_selected_models,
        inputs=[image_input, model_selector],
        outputs=image_outputs
    )

demo.launch()
