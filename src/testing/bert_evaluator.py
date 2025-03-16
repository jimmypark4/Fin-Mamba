# src/testing/bert_evaluator.py
import torch
import logging
from .base_evaluator import BaseEvaluator

class BertEvaluator(BaseEvaluator):
    def forward_test(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        token_type_ids = batch["token_type_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        # 모델 Forward (예: self.model(...)에 맞게 인자명 일치)
        logits = self.model(
            bert_sent=input_ids,
            bert_sent_type=token_type_ids,
            bert_sent_mask=attention_mask
        )

        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()
        total = labels.size(0)

        # (NEW) CPU list로 변환 -> F1 계산용
        preds_list = preds.detach().cpu().tolist()
        labels_list = labels.detach().cpu().tolist()

        # (loss, correct, total, preds, labels)
        return loss.item(), correct, total, preds_list, labels_list
