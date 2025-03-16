import torch
import torch.nn as nn
import logging
from tqdm import tqdm
import os

class BaseTrainer:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            logging.info("Using GPU")
        else:
            logging.info("Using CPU")

        self.model.to(self.device)

        # 체크포인트 저장 디렉토리
        self.checkpoint_dir = cfg.experiment.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Optimizer & Loss
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

        self.epochs = cfg.training.epochs

        # Early Stopping 관련 변수
        self.early_stopping_patience = cfg.training.early_stopping_patience  # 연속 상승 허용 횟수
        self.early_stopping_counter = 0  # 연속 상승 카운터
        self.best_loss = float('inf')  # 최소 loss 추적

    def train(self, train_loader, valid_loader=None):
        """
        공통 학습 루프 로직.
        각 미니배치마다 self.forward(batch)를 호출해 loss 계산 후 backward.
        """
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                self.optimizer.zero_grad()
                loss = self.forward(batch)  # <-- 자식 클래스에서 구현
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            logging.info(f"[Epoch {epoch+1}/{self.epochs}] Train Loss: {avg_loss:.4f}")

            # Validation (옵션)
            if valid_loader is not None:
                val_result = self.validate(valid_loader)
                logging.info(f"Val Loss: {val_result['loss']:.4f}, Val Acc: {val_result['acc']*100:.2f}%")

            # Early Stopping 체크
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                if self.early_stopping_counter > 0:
                    logging.info(f"stopping counter -> ({self.early_stopping_counter}  )")
                else:
                    self.early_stopping_counter = 0
                    # logging.info(f"stopping counter -> ({self.early_stopping_counter}  )")///
            else:
                self.early_stopping_counter += 1
                logging.info(f"Early Stopping Counter: {self.early_stopping_counter}/{self.early_stopping_patience}")

                if self.early_stopping_counter >= self.early_stopping_patience:
                    logging.info("Early stopping triggered. Training terminated.")
                    break

            # Epoch이 끝날 때마다 체크포인트 저장 (필요 시 활성화)
            # checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch+1}_model.pt")
            # torch.save(self.model.state_dict(), checkpoint_path)
            # logging.info(f"Checkpoint saved at {checkpoint_path}")

    def validate(self, valid_loader):
        """
        공통 검증 루프 로직.
        각 미니배치마다 self.forward_validation(batch)를 호출해 loss 및 acc 계산.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in valid_loader:
                loss_val, c, t = self.forward_validation(batch)  # <-- 자식 클래스에서 구현
                total_loss += loss_val
                correct += c
                total += t

        avg_loss = total_loss / len(valid_loader)
        accuracy = correct / total if total > 0 else 0
        return {"loss": avg_loss, "acc": accuracy}

    def forward(self, batch):
        """
        학습용 forward 메서드 (loss 계산).
        자식 클래스에서 구체적으로 구현해야 함.
        """
        raise NotImplementedError("forward() must be overridden in child class.")

    def forward_validation(self, batch):
        """
        검증용 forward 메서드 (loss + acc 계산).
        자식 클래스에서 구체적으로 구현해야 함.
        """
        raise NotImplementedError("forward_validation() must be overridden in child class.")
