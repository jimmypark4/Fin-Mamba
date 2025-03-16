import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType

# FinBERT 모델에 LoRA를 적용한 PEFT 모델 (AutoModelForSequenceClassification 사용)
class FinBERT_LoRA(nn.Module):
    def __init__(self, config):
        """
        config:
          - config.embedding: FinBERT 모델 경로 (예: "ProsusAI/finbert")
          - config.num_labels: 분류 클래스 수
          - config.lora_r: LoRA 저랭크 차원 (예: 8)
          - config.lora_alpha: LoRA alpha 값 (예: 32)
          - config.lora_dropout: LoRA dropout 비율 (예: 0.2)
          - config.target_modules: LoRA 적용 대상 모듈 리스트 (예: ["query", "value"])
        """
        super().__init__()
        self.num_labels = config.num_labels
        self.embedding = config.embedding

        # 사전학습된 FinBERT 분류 모델 로드 (labels 인자도 처리 가능)
        self.config_model = AutoConfig.from_pretrained(self.embedding, output_hidden_states=True)
        self.finetuned_model = AutoModelForSequenceClassification.from_pretrained(
            self.embedding,
            config=self.config_model
        )

        # 전체 모델 파라미터 Freeze
        for param in self.finetuned_model.parameters():
            param.requires_grad = False

        # LoRA 설정
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules
        )

        # LoRA 적용: 기본 모델에 LoRA 파라미터 추가
        self.lora_model = get_peft_model(self.finetuned_model, lora_config)

        # 분류 헤드는 반드시 학습 가능하도록 unfreeze
        for name, param in self.lora_model.named_parameters():
            if "classifier" in name:
                param.requires_grad = True

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.lora_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return outputs.logits
