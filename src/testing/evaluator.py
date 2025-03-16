# src/testing/evaluator.py
import torch
import logging
class Evaluator:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def evaluate(self, test_loader):
        self.model.eval()
        total = 0
        correct = 0

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(
                    bert_sent=input_ids,
                    bert_sent_type=token_type_ids,
                    bert_sent_mask=attention_mask
                )

                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0
        logging.info("\n\n\n\n\n")
        logging.info(f"Test Acc: {accuracy*100:.2f}%")
        logging.info("\n\n\n\n\n")

        return {"accuracy": accuracy}
