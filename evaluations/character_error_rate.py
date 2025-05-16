import torch
from torchmetrics import Metric
from ocr import OCR
import Levenshtein


class CharacterErrorRate(Metric):
    def __init__(self, ocr, dist_sync_on_step=False):
        # super().__init__(dist_sync_on_step=dist_sync_on_step)
        super().__init__()
        self.add_state("total_errors", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_chars", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.ocr = ocr

    def update(self, pred_images, target_images):
        for pred_img, target_img in zip(pred_images, target_images):
            pred_text = self.ocr.predict(pred_img)
            target_text = self.ocr.predict(target_img)

            dist = Levenshtein.distance(pred_text, target_text)
            self.total_errors += dist
            self.total_chars += len(target_text)

    def compute(self):
        if self.total_chars == 0:
            return torch.tensor(0.0)
        return self.total_errors / self.total_chars
