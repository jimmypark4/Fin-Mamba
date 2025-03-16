import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType

# FinBERT 모델에 qLoRA를 적용한 PEFT 모델 (분류 헤드 unfreeze 포함)
class FinBERT_qLoRA(nn.Module):
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

        # BitsAndBytesConfig를 사용하여 8비트 양자화 설정
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # 사전학습된 FinBERT 분류 모델 로드 (hidden_states 포함)
        self.config_model = AutoConfig.from_pretrained(self.embedding, output_hidden_states=True)
        # device_map은 예시로 GPU 0에 전체 모델을 할당합니다.
        self.finetuned_model = AutoModelForSequenceClassification.from_pretrained(
            self.embedding,
            config=self.config_model,
            quantization_config=quantization_config,
            device_map={"": 0}
        )

        # Freeze: 전체 모델 파라미터 동결
        for param in self.finetuned_model.parameters():
            param.requires_grad = False

        # LoRA 설정 (qLoRA 적용)
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules
        )

        # qLoRA 적용 (양자화된 모델에 LoRA 파라미터 추가)
        self.q_lora_model = get_peft_model(self.finetuned_model, lora_config)

        # 분류 헤드는 반드시 학습 가능하도록 unfreeze (qLoRA 모델 내 분류 헤드)
        for name, param in self.q_lora_model.named_parameters():
            if "classifier" in name:
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.q_lora_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
