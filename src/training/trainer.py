# src/training/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
class Trainer:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            logging.info("Using GPU")
        self.model.to(self.device)

        # Hydra 설정에서 체크포인트 디렉토리를 가져옴

        self.checkpoint_dir = cfg.experiment.checkpoint_dir

        # Optimizer & Loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.training.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = cfg.training.epochs

    def train(self, train_loader, valid_loader=None):
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(
                    bert_sent=input_ids,
                    bert_sent_type=token_type_ids,
                    bert_sent_mask=attention_mask
                )

                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            if (epoch+1) % 10 == 0:
                logging.info(f"[Epoch {epoch+1}/{self.epochs}] "f"Train Loss: {avg_loss:.4f}")


            # Validation 생략하거나, 필요 시 valid_loader 사용 가능
            if valid_loader is not None:
                self.validate(valid_loader)

        # 마지막 체크포인트 저장
        torch.save(self.model, f"{self.checkpoint_dir}/epoch_{epoch+1}_model.pt")

    def validate(self, valid_loader):
        self.model.eval()
        loss_arr = []
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(
                    bert_sent=input_ids,
                    bert_sent_type=token_type_ids,
                    bert_sent_mask=attention_mask
                )

                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(valid_loader)
        accuracy = correct / total if total > 0 else 0
        print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%")
        return {"loss": avg_loss, "acc": accuracy}
