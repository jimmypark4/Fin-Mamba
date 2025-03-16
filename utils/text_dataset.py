from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    """
    BERT 입력 형식에 맞게 (input_ids, attention_mask, token_type_ids, label)을 구성
    """
    def __init__(self, df):
        super().__init__()
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()

        # bert-base-uncased 토크나이저
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=32,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)         # (seq_len)
        attention_mask = encoding["attention_mask"].squeeze(0)
        token_type_ids = encoding["token_type_ids"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": torch.tensor(label, dtype=torch.long)
        }