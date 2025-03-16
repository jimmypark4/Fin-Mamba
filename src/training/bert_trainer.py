# src/training/bert_trainer.py
import torch
from .base_trainer import BaseTrainer

class BertTrainer(BaseTrainer):
    """
    BERT 기반 모델을 위한 Trainer
    """
    def forward(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        token_type_ids = batch["token_type_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        # --- 변경: 모델 인자명을 'bert_sent', 'bert_sent_type', 'bert_sent_mask' 로 맞춤 ---
        logits = self.model(
            bert_sent=input_ids,
            bert_sent_type=token_type_ids,
            bert_sent_mask=attention_mask
        )
        loss = self.criterion(logits, labels)
        return loss

    def forward_validation(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        token_type_ids = batch["token_type_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        # --- 동일 변경 ---
        logits = self.model(
            bert_sent=input_ids,
            bert_sent_type=token_type_ids,
            bert_sent_mask=attention_mask
        )
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()
        total = labels.size(0)

        return loss.item(), correct, total
