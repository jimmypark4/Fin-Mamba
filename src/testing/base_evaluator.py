# src/testing/base_evaluator.py

import torch
import logging
from torch.nn import CrossEntropyLoss
from sklearn.metrics import f1_score

class BaseEvaluator:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = CrossEntropyLoss()

    def evaluate(self, test_loader):
        """
        공통 평가 로직:
        - test_loader 순회하며 forward_test()를 호출해 loss, correct, total,
          preds, labels를 누적
        - 최종 avg_loss, accuracy, f1 반환
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        # (NEW) F1 계산 위해 배치별 예측/정답 저장
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                # 자식 클래스에서 (loss_val, c, t, preds, labels) 형태로 반환
                loss_val, c, t, preds, labels = self.forward_test(batch)

                total_loss += loss_val
                correct += c
                total += t

                all_preds.extend(preds)
                all_labels.extend(labels)

        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
        accuracy = correct / total if total > 0 else 0

        # (NEW) F1 계산 (다중분류이므로 average="macro" or "weighted" 등 적절히)
        f1_val = 0.0
        if len(all_labels) > 0:
            f1_val = f1_score(all_labels, all_preds, average="macro")

        logging.info("\n\n\n\n\n")
        logging.info(f"[Evaluator] Test Loss: {avg_loss:.4f}, Test Acc: {accuracy*100:.2f}%, F1: {f1_val:.4f}")
        logging.info("\n\n\n\n\n")

        return {"loss": avg_loss, "accuracy": accuracy, "f1": f1_val}

    def forward_test(self, batch):
        """
        테스트용 forward (loss + acc + preds + labels) 계산.
        자식 클래스에서 오버라이드.
        반드시 (loss_val, correct, total, preds, labels) 형태로 반환하도록 변경.
        """
        raise NotImplementedError("forward_test() must be overridden in child class.")
