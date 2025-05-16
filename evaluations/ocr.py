from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch


class OCR:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.model = AutoModelForImageTextToText.from_pretrained(
            "google/gemma-3-12b-it",
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained("google/gemma-3-12b-it")

        self.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": "Extract and output only the text from the image in its original language. If there is no text, return nothing.",
                    },
                ],
            },
        ]

    def predict(self, image):
        image = (
            (image * 255).clamp(0, 255).to(torch.uint8).permute((1, 2, 0)).cpu().numpy()
        )
        image = Image.fromarray(image).convert("RGB").resize((1024, 1024))
        prompt = self.processor.apply_chat_template(
            self.messages, add_generation_prompt=True
        )
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[-1] :], skip_special_tokens=True
        )[0]
        return generated_text
