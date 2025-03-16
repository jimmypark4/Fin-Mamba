import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class FinBertTextDataset(Dataset):
    """
    FinBERT 입력 형식에 맞게 (input_ids, attention_mask, token_type_ids, label)을 구성
    """
    def __init__(self, df, max_length=32):
        super().__init__()
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()

        # FinBERT 토크나이저
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        # shape: (1, seq_len)이므로 squeeze(0)로 (seq_len)으로 변환
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        token_type_ids = encoding["token_type_ids"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": torch.tensor(label, dtype=torch.long)
        }
