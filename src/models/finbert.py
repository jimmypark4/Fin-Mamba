import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
# 필요한 경우 AutoTokenizer도 임포트하세요.
# from transformers import AutoTokenizer

class FinBERT(nn.Module):
    def __init__(self, config):
        """
        config 예시:
          config.num_labels = 3  (기본 3-class 분류)
        """
        super().__init__()
        self.num_labels = getattr(config, "num_labels", 3)

        # 1) FinBERT 로드 (Pretrained)
        #    - AutoConfig를 통해 output_hidden_states=True 설정
        self.finbert_config = AutoConfig.from_pretrained(
            "ProsusAI/finbert",
            output_hidden_states=True
        )
        self.finbert_model = AutoModel.from_pretrained(
            "ProsusAI/finbert",
            config=self.finbert_config
        )

        # 2) 분류 레이어 (FinBERT hidden_size = 768)
        self.fc = nn.Linear(768, self.num_labels)

    def forward(self, input_ids, attention_mask):
        """
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)

        반환값:
          logits: (batch_size, self.num_labels)
        """
        # (a) FinBERT 인코딩
        outputs = self.finbert_model(
            input_ids=input_ids,
            attention_mask=attention_mask
            # token_type_ids 등을 사용하고 싶다면 추가로 전달.
        )
        # BERT의 마지막 레이어 출력: (batch_size, seq_len, hidden_size=768)
        sequence_output = outputs.last_hidden_state

        # (b) attention_mask 기반 마스킹 후 평균 풀링
        masked_output = torch.mul(sequence_output, attention_mask.unsqueeze(2))  # (batch_size, seq_len, 768)
        mask_len = torch.sum(attention_mask, dim=1, keepdim=True)               # (batch_size, 1)
        pooled_output = torch.sum(masked_output, dim=1) / mask_len              # (batch_size, 768)

        # (c) 분류 레이어
        logits = self.fc(pooled_output)  # (batch_size, num_labels)
        return logits
# 모델 초기화 및 파라미터 수 계산
if __name__ == "__main__":
    from parameter_counter import count_parameters
    from omegaconf import DictConfig

    # 예시 Config 정의
    config = DictConfig({

    })

    # 모델 생성
    model = FinBERT(config)

    # 파라미터 수 계산
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")  # 총 파라미터 수를 출력
    