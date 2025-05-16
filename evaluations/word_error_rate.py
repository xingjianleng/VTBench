import torch
from torchmetrics import Metric
import Levenshtein


class WordErrorRate(Metric):
    def __init__(self, ocr, dist_sync_on_step=False):
        # super().__init__(dist_sync_on_step=dist_sync_on_step)
        super().__init__()
        self.ocr = ocr
        self.add_state("total_errors", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_words", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, pred_images, target_images):
        for pred_img, target_img in zip(pred_images, target_images):
            pred_text = self.ocr.predict(pred_img)
            target_text = self.ocr.predict(target_img)

            pred_words = pred_text.strip().split()
            target_words = target_text.strip().split()

            dist = Levenshtein.distance(" ".join(pred_words), " ".join(target_words))

            self.total_errors += dist
            self.total_words += len(target_words)

    def compute(self):
        if self.total_words == 0:
            return torch.tensor(0.0)
        return self.total_errors / self.total_words
