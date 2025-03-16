# src/testing/mamba_evaluator.py
import torch
import logging
from .base_evaluator import BaseEvaluator

class MambaEvaluator(BaseEvaluator):
    def forward_test(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()
        total = labels.size(0)

        preds_list = preds.detach().cpu().tolist()
        labels_list = labels.detach().cpu().tolist()

        return loss.item(), correct, total, preds_list, labels_list
