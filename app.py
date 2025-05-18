import gradio as gr
from src.data_processing import pil_to_tensor, tensor_to_pil
from PIL import Image
from src.model_processing import get_model
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

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
    model, data_params = get_model("./VTBench_models", model_name)
    model = model.to(device)
    model.eval()
    return model, data_params

model_dict = {
    model_name: load_model(model_name)
    for model_name in model_name_mapping
}

placeholder_image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))

def process_selected_models(uploaded_image, selected_models):
    results = []
    for model_name in model_name_mapping:
        if uploaded_image is None:
            results.append(gr.update(value=placeholder_image, label=f"{model_name_mapping[model_name]} (No input)"))
        elif model_name in selected_models:
            try:
                model, data_params = model_dict[model_name]
                pixel_values = pil_to_tensor(uploaded_image, **data_params).unsqueeze(0).to(device)
                output = model(pixel_values)[0]
                reconstructed_image = tensor_to_pil(output[0].cpu(), **data_params)
                results.append(gr.update(value=reconstructed_image, label=model_name_mapping[model_name]))
            except Exception as e:
                print(f"Error in model {model_name}: {e}")
                results.append(gr.update(value=placeholder_image, label=f"{model_name_mapping[model_name]} (Error)"))
        else:
            results.append(gr.update(value=placeholder_image, label=f"{model_name_mapping[model_name]} (Not selected)"))
    return results

with gr.Blocks() as demo:
    gr.Markdown("## VTBench")

    gr.Markdown("---")

    image_input = gr.Image(type="pil", label="Upload an image")

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
    model_items = list(model_name_mapping.items())
    output_rows = [model_items[i:i+3] for i in range(0, len(model_items), 3)]

    with gr.Column():
        for row in output_rows:
            with gr.Row():
                for model_name, display_name in row:
                    out_img = gr.Image(
                        label=f"{display_name} (Not run)",
                        value=placeholder_image
                    )
                    image_outputs.append(out_img)

    run_button.click(
        fn=process_selected_models,
        inputs=[image_input, model_selector],
        outputs=image_outputs
    )

demo.launch()
