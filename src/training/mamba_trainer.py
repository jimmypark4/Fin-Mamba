# src/training/mamba_trainer.py
import torch
from .base_trainer import BaseTrainer

class MambaTrainer(BaseTrainer):
    """
    Mamba(State-Space) 기반 모델을 위한 Trainer
    - forward에서 token_type_ids 없이 input_ids, attention_mask만 사용
    """
    def forward(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        loss = self.criterion(logits, labels)
        return loss

    def forward_validation(self, batch):
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

        return loss.item(), correct, total
