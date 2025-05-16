import torch
import torch.nn.functional as F
import lightning as L

from contextlib import contextmanager
from collections import OrderedDict

from .improved_model import Encoder, Decoder
from .lookup_free_quantize import LFQ
from .ema import LitEma


class VQModel(L.LightningModule):
    def __init__(
        self,
        ddconfig,
        lossconfig,
        ## Quantize Related
        n_embed,
        embed_dim,
        sample_minimization_weight,
        batch_maximization_weight,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        learning_rate=None,
        resume_lr=None,
        ### scheduler config
        warmup_epochs=1.0,  # warmup epochs
        scheduler_type="linear-warmup_cosine-decay",
        min_learning_rate=0,
        use_ema=False,
        token_factorization=False,
        stage=None,
        lr_drop_epoch=None,
        lr_drop_rate=0.1,
        factorized_bits=[9, 9],
    ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = LFQ(
            dim=embed_dim,
            codebook_size=n_embed,
            sample_minimization_weight=sample_minimization_weight,
            batch_maximization_weight=batch_maximization_weight,
            token_factorization=token_factorization,
            factorized_bits=factorized_bits,
        )

        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.use_ema = use_ema
        if (
            self.use_ema and stage is None
        ):  # no need to construct EMA when training Transformer
            self.model_ema = LitEma(self)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, stage=stage)
        self.resume_lr = resume_lr
        self.learning_rate = learning_rate
        self.lr_drop_epoch = lr_drop_epoch
        self.lr_drop_rate = lr_drop_rate
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.min_learning_rate = min_learning_rate
        self.automatic_optimization = False

        self.strict_loading = False

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def load_state_dict(self, *args, strict=False):
        """
        Resume not strict loading
        """
        return super().load_state_dict(*args, strict=strict)

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        """
        filter out the non-used keys
        """
        return {
            k: v
            for k, v in super()
            .state_dict(*args, destination, prefix, keep_vars)
            .items()
            if (
                "inception_model" not in k
                and "lpips_vgg" not in k
                and "lpips_alex" not in k
            )
        }

    def init_from_ckpt(self, path, ignore_keys=list(), stage="transformer"):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        ema_mapping = {}
        new_params = OrderedDict()
        if stage == "transformer":  ### directly use ema encoder and decoder parameter
            if self.use_ema:
                for k, v in sd.items():
                    if "encoder" in k:
                        if "model_ema" in k:
                            k = k.replace(
                                "model_ema.", ""
                            )  # load EMA Encoder or Decoder
                            new_k = ema_mapping[k]
                            new_params[new_k] = v
                        s_name = k.replace(".", "")
                        ema_mapping.update({s_name: k})
                        continue
                    if "decoder" in k:
                        if "model_ema" in k:
                            k = k.replace(
                                "model_ema.", ""
                            )  # load EMA Encoder or Decoder
                            new_k = ema_mapping[k]
                            new_params[new_k] = v
                        s_name = k.replace(".", "")
                        ema_mapping.update({s_name: k})
                        continue
            else:  # also only load the Generator
                for k, v in sd.items():
                    if "encoder" in k:
                        new_params[k] = v
                    elif "decoder" in k:
                        new_params[k] = v
        missing_keys, unexpected_keys = self.load_state_dict(
            new_params, strict=False
        )  # first stage
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        (quant, emb_loss, info), loss_breakdown = self.quantize(
            h, return_loss_breakdown=True
        )
        return quant, emb_loss, info, loss_breakdown

    def decode(self, quant):
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, img_toks, loss_break = self.encode(input)
        pixels = self.decode(quant)
        return pixels, img_toks, quant

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).contiguous()
        return x.float()

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x
