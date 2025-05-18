import requests
import os
import yaml
from .utils import get_ckpt, get_yaml_config


def download_ckpt_yaml(model_path, model_name, ckpt_path, yaml_url=None):
    def download_file(url, save_path):
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)

    # os.makedirs(model_path, exist_ok=True)
    local_dir = os.path.join(model_path, model_name)
    os.makedirs(local_dir, exist_ok=True)

    ckpt_name = ckpt_path.split("/")[-1]
    local_ckpt_path = os.path.join(local_dir, ckpt_name)
    if not os.path.exists(local_ckpt_path):
        print(f"Downloading CKPT to {local_ckpt_path}")
        download_file(ckpt_path, local_ckpt_path)

    if yaml_url:
        yaml_name = yaml_url.split("/")[-1]
        local_yaml_path = os.path.join(local_dir, yaml_name)
        if not os.path.exists(local_yaml_path):
            print(f"Downloading YAML to {local_yaml_path}")
            download_file(yaml_url, local_yaml_path)
        return local_ckpt_path, local_yaml_path

    return local_ckpt_path, None


def get_model(model_path, model_name):
    model = None
    data_params = {
        "target_image_size": (512, 512),
        "lock_ratio": True,
        "center_crop": True,
        "padding": False,
    }

    if model_name.lower() == "anole":
        from src.vqvaes.anole.anole import VQModel
        yaml_url = "https://huggingface.co/GAIR/Anole-7b-v0.1/resolve/main/tokenizer/vqgan.yaml"
        ckpt_path = "https://huggingface.co/GAIR/Anole-7b-v0.1/resolve/main/tokenizer/vqgan.ckpt"

        if model_path is not None:
            ckpt_path, yaml_url = download_ckpt_yaml(model_path, "anole", ckpt_path, yaml_url)
        config = get_yaml_config(yaml_url)

        params = config["model"]["params"]
        if "lossconfig" in params:
            del params["lossconfig"]
        params["ckpt_path"] = ckpt_path
        model = VQModel(**params)
        data_params = {
            "target_image_size": (512, 512),
            "lock_ratio": True,
            "center_crop": True,
            "padding": False,
        }

    elif model_name.lower() == "chameleon":
        from src.vqvaes.anole.anole import VQModel

        yaml_url = "https://huggingface.co/huaweilin/chameleon_vqvae/resolve/main/vqgan.yaml"
        ckpt_path = "https://huggingface.co/huaweilin/chameleon_vqvae/resolve/main/vqgan.ckpt"
        if model_path is not None:
            ckpt_path, yaml_url = download_ckpt_yaml(model_path, "chameleon", ckpt_path, yaml_url)
        config = get_yaml_config(yaml_url)

        params = config["model"]["params"]
        if "lossconfig" in params:
            del params["lossconfig"]
        params["ckpt_path"] = ckpt_path
        model = VQModel(**params)
        data_params = {
            "target_image_size": (512, 512),
            "lock_ratio": True,
            "center_crop": True,
            "padding": False,
        }

    elif model_name.lower() == "llamagen-ds16":
        from src.vqvaes.llamagen.llamagen import VQ_models
        ckpt_path = "https://huggingface.co/FoundationVision/LlamaGen/resolve/main/vq_ds16_c2i.pt"
        if model_path is not None:
            ckpt_path, _ = download_ckpt_yaml(model_path, "llamagen-ds16", ckpt_path, None)

        model = VQ_models["VQ-16"](codebook_size=16384, codebook_embed_dim=8)
        model.load_state_dict(get_ckpt(ckpt_path, key="model"))
        data_params = {
            "target_image_size": (512, 512),
            "lock_ratio": True,
            "center_crop": True,
            "padding": False,
        }

    elif model_name.lower() == "llamagen-ds16-t2i":
        from src.vqvaes.llamagen.llamagen import VQ_models
        ckpt_path = "https://huggingface.co/peizesun/llamagen_t2i/resolve/main/vq_ds16_t2i.pt"
        if model_path is not None:
            ckpt_path, _ = download_ckpt_yaml(model_path, "llamagen-ds16-t2i", ckpt_path, None)

        model = VQ_models["VQ-16"](codebook_size=16384, codebook_embed_dim=8)
        model.load_state_dict(get_ckpt(ckpt_path, key="model"))
        data_params = {
            "target_image_size": (512, 512),
            "lock_ratio": True,
            "center_crop": True,
            "padding": False,
        }

    elif model_name.lower() == "llamagen-ds8":
        from src.vqvaes.llamagen.llamagen import VQ_models
        ckpt_path = "https://huggingface.co/FoundationVision/LlamaGen/resolve/main/vq_ds8_c2i.pt"
        if model_path is not None:
            ckpt_path, _ = download_ckpt_yaml(model_path, "llamagen-ds8", ckpt_path, None)

        model = VQ_models["VQ-8"](codebook_size=16384, codebook_embed_dim=8)
        model.load_state_dict(get_ckpt(ckpt_path, key="model"))
        data_params = {
            "target_image_size": (256, 256),
            "lock_ratio": True,
            "center_crop": True,
            "padding": False,
        }

    elif model_name.lower() == "flowmo_lo":
        from src.vqvaes.flowmo.flowmo import build_model
        yaml_url = "https://raw.githubusercontent.com/kylesargent/FlowMo/refs/heads/main/flowmo/configs/base.yaml"
        ckpt_path = "https://huggingface.co/ksarge/FlowMo/resolve/main/flowmo_lo.pth"
        if model_path is not None:
            ckpt_path, yaml_url = download_ckpt_yaml(model_path, "flowmo_lo", ckpt_path, yaml_url)
        config = get_yaml_config(yaml_url)

        config.model.context_dim = 18
        model = build_model(config)
        model.load_state_dict(
            get_ckpt(ckpt_path, key="model_ema_state_dict")
        )
        data_params = {
            "target_image_size": (256, 256),
            "lock_ratio": True,
            "center_crop": True,
            "padding": False,
        }

    elif model_name.lower() == "flowmo_hi":
        from src.vqvaes.flowmo.flowmo import build_model

        yaml_url = "https://raw.githubusercontent.com/kylesargent/FlowMo/refs/heads/main/flowmo/configs/base.yaml"
        ckpt_path = "https://huggingface.co/ksarge/FlowMo/resolve/main/flowmo_hi.pth"
        if model_path is not None:
            ckpt_path, yaml_url = download_ckpt_yaml(model_path, "flowmo_hi", ckpt_path, yaml_url)
        config = get_yaml_config(yaml_url)

        config.model.context_dim = 56
        config.model.codebook_size_for_entropy = 14
        model = build_model(config)
        model.load_state_dict(
            get_ckpt(ckpt_path, key="model_ema_state_dict")
        )
        data_params = {
            "target_image_size": (256, 256),
            "lock_ratio": True,
            "center_crop": True,
            "padding": False,
        }

    elif model_name.lower() == "open_magvit2":
        from src.vqvaes.open_magvit2.open_magvit2 import VQModel

        yaml_url = "https://raw.githubusercontent.com/TencentARC/SEED-Voken/refs/heads/main/configs/Open-MAGVIT2/gpu/imagenet_lfqgan_256_L.yaml"
        ckpt_path = "https://huggingface.co/TencentARC/Open-MAGVIT2-Tokenizer-256-resolution/resolve/main/imagenet_256_L.ckpt"
        if model_path is not None:
            ckpt_path, yaml_url = download_ckpt_yaml(model_path, "open_magvit2", ckpt_path, yaml_url)
        config = get_yaml_config(yaml_url)

        model = VQModel(**config.model.init_args)
        model.load_state_dict(get_ckpt(ckpt_path, key="state_dict"))
        data_params = {
            "target_image_size": (256, 256),
            "lock_ratio": True,
            "center_crop": True,
            "padding": False,
        }

    elif "maskbit" in model_name.lower():
        from src.vqvaes.maskbit.maskbit import ConvVQModel

        if "16bit" in model_name.lower():
            yaml_url = "https://raw.githubusercontent.com/markweberdev/maskbit/refs/heads/main/configs/tokenizer/maskbit_tokenizer_16bit.yaml"
            ckpt_path = "https://huggingface.co/markweber/maskbit_tokenizer_16bit/resolve/main/maskbit_tokenizer_16bit.bin"
            if model_path is not None:
                ckpt_path, yaml_url = download_ckpt_yaml(model_path, "maskbit-16bit", ckpt_path, yaml_url)
        elif "18bit" in model_name.lower():
            yaml_url = "https://raw.githubusercontent.com/markweberdev/maskbit/refs/heads/main/configs/tokenizer/maskbit_tokenizer_18bit.yaml"
            ckpt_path = "https://huggingface.co/markweber/maskbit_tokenizer_18bit/resolve/main/maskbit_tokenizer_18bit.bin"
            if model_path is not None:
                ckpt_path, yaml_url = download_ckpt_yaml(model_path, "maskbit-18bit", ckpt_path, yaml_url)
        else:
            raise Exception(f"Unsupported model: {model_name}")

        config = get_yaml_config(yaml_url)
        model = ConvVQModel(config.model.vq_model, legacy=False)
        model.load_pretrained(get_ckpt(ckpt_path, key=None))
        data_params = {
            "target_image_size": (256, 256),
            "lock_ratio": True,
            "center_crop": True,
            "padding": False,
            "standardize": False,
        }

    elif "bsqvit" in model_name.lower():
        from src.vqvaes.bsqvit.bsqvit import VITBSQModel

        yaml_url = "https://huggingface.co/huaweilin/bsqvit_256x256/resolve/main/config.yaml"
        ckpt_path = "https://huggingface.co/huaweilin/bsqvit_256x256/resolve/main/checkpoint.pt"
        if model_path is not None:
            ckpt_path, yaml_url = download_ckpt_yaml(model_path, "bsqvit", ckpt_path, yaml_url)

        config = get_yaml_config(yaml_url)
        model = VITBSQModel(**config["model"]["params"])
        model.init_from_ckpt(get_ckpt(ckpt_path, key="state_dict"))
        data_params = {
            "target_image_size": (256, 256),
            "lock_ratio": True,
            "center_crop": True,
            "padding": False,
            "standardize": False,
        }

    elif "titok" in model_name.lower():
        from src.vqvaes.titok.titok import TiTok

        ckpt_path = None
        if "bl64" in model_name.lower():
            ckpt_path = "yucornetto/tokenizer_titok_bl64_vq8k_imagenet"
        elif "bl128" in model_name.lower():
            ckpt_path = "yucornetto/tokenizer_titok_bl128_vq8k_imagenet"
        elif "sl256" in model_name.lower():
            ckpt_path = "yucornetto/tokenizer_titok_sl256_vq8k_imagenet"
        elif "l32" in model_name.lower():
            ckpt_path = "yucornetto/tokenizer_titok_l32_imagenet"
        elif "b64" in model_name.lower():
            ckpt_path = "yucornetto/tokenizer_titok_b64_imagenet"
        elif "s128" in model_name.lower():
            ckpt_path = "yucornetto/tokenizer_titok_s128_imagenet"
        else:
            raise Exception(f"Unsupported model: {model_name}")

        model = TiTok.from_pretrained(ckpt_path)
        data_params = {
            "target_image_size": (256, 256),
            "lock_ratio": True,
            "center_crop": True,
            "padding": False,
            "standardize": False,
        }

    elif "janus_pro" in model_name.lower():
        from janus.models import MultiModalityCausalLM
        from src.vqvaes.janus_pro.janus_pro import forward
        import types

        model = MultiModalityCausalLM.from_pretrained(
            "deepseek-ai/Janus-Pro-7B", trust_remote_code=True
        ).gen_vision_model
        model.forward = types.MethodType(forward, model)
        data_params = {
            "target_image_size": (384, 384),
            "lock_ratio": True,
            "center_crop": False,
            "padding": True,
        }

    elif "var" in model_name.lower():
        from src.vqvaes.var.var_vq import VQVAE

        ckpt_path = "https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth"
        if model_path is not None:
            ckpt_path, _ = download_ckpt_yaml(model_path, "var", ckpt_path, None)

        v_patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        if "512" in model_name.lower():
            v_patch_nums = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32)
        model = VQVAE(
            vocab_size=4096,
            z_channels=32,
            ch=160,
            test_mode=True,
            share_quant_resi=4,
            v_patch_nums=v_patch_nums,
        )
        model.load_state_dict(get_ckpt(ckpt_path, key=None))
        data_params = {
            "target_image_size": (
                (512, 512) if "512" in model_name.lower() else (256, 256)
            ),
            "lock_ratio": True,
            "center_crop": False,
            "padding": True,
            "standardize": False,
        }

    elif (
        "infinity" in model_name.lower()
    ):  # "infinity_d32", "infinity_d64", "infinity_d56_f8_14_patchify"
        from src.vqvaes.infinity.vae import vae_model

        if "d32" in model_name:
            ckpt_path = "https://huggingface.co/FoundationVision/Infinity/resolve/main/infinity_vae_d32.pth"
            codebook_dim = 32
            if model_path is not None:
                ckpt_path, _ = download_ckpt_yaml(model_path, "infinity-d32", ckpt_path, None)
        elif "d64" in model_name:
            ckpt_path = "https://huggingface.co/FoundationVision/Infinity/resolve/main/infinity_vae_d64.pth"
            codebook_dim = 64
            if model_path is not None:
                ckpt_path, _ = download_ckpt_yaml(model_path, "infinity-d64", ckpt_path, None)

        schedule_mode = "dynamic"
        codebook_size = 2**codebook_dim
        patch_size = 16
        encoder_ch_mult = [1, 2, 4, 4, 4]
        decoder_ch_mult = [1, 2, 4, 4, 4]

        ckpt = get_ckpt(ckpt_path, key=None)
        model = vae_model(
            ckpt,
            schedule_mode,
            codebook_dim,
            codebook_size,
            patch_size=patch_size,
            encoder_ch_mult=encoder_ch_mult,
            decoder_ch_mult=decoder_ch_mult,
            test_mode=True,
        )

        data_params = {
            "target_image_size": (1024, 1024),
            "lock_ratio": True,
            "center_crop": False,
            "padding": True,
            "standardize": False,
        }

    elif "sd3.5l" in model_name.lower():  # SD3.5L
        from src.vaes.stable_diffusion.vae import forward
        from diffusers import AutoencoderKL
        import types

        model = AutoencoderKL.from_pretrained(
            "huaweilin/stable-diffusion-3.5-large-vae", subfolder="vae"
        )
        model.forward = types.MethodType(forward, model)
        data_params = {
            "target_image_size": (1024, 1024),
            "lock_ratio": True,
            "center_crop": False,
            "padding": True,
            "standardize": True,
        }

    elif "FLUX.1-dev".lower() in model_name.lower():  # SD3.5L
        from src.vaes.stable_diffusion.vae import forward
        from diffusers import AutoencoderKL
        import types

        model = AutoencoderKL.from_pretrained(
            "black-forest-labs/FLUX.1-dev", subfolder="vae"
        )
        model.forward = types.MethodType(forward, model)
        data_params = {
            "target_image_size": (1024, 1024),
            "lock_ratio": True,
            "center_crop": False,
            "padding": True,
            "standardize": True,
        }

    elif "gpt4o" in model_name.lower():
        from src.vaes.gpt_image.gpt_image import GPTImage

        data_params = {
            "target_image_size": (1024, 1024),
            "lock_ratio": True,
            "center_crop": False,
            "padding": True,
            "standardize": False,
        }
        model = GPTImage(data_params)

    else:
        raise Exception(f"Unsupported model: \"{model_name}\"")

    try:
        trainable_params = sum(p.numel() for p in model.parameters())
        print("trainable_params:", trainable_params)
    except Exception as e:
        print(e)
        pass

    model.eval()
    return model, data_params
