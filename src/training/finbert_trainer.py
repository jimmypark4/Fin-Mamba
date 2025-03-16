# src/training/finbert_trainer.py
import torch
from .base_trainer import BaseTrainer

class FinBertTrainer(BaseTrainer):
    """
    FinBERT 모델 학습을 위한 Trainer
    - BaseTrainer를 상속받아, forward와 forward_validation을 구현
    """
    def forward(self, batch):
        """
        훈련(Train) 시에 호출되어, 손실값을 반환
        """
        # 1) 배치로부터 필요한 텐서 추출 후 GPU/CPU 할당
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # 2) FinBERT 모델 순전파(Forward)
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 3) 손실 계산
        loss = self.criterion(logits, labels)
        return loss

    def forward_validation(self, batch):
        """
        검증(Validation) 시에 호출되어, (loss, correct, total)을 반환
        """
        # 1) 배치 → device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # 2) FinBERT 모델 Forward
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 3) 손실 + 정확도 계산
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()
        total = labels.size(0)

        return loss.item(), correct, total
