import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
"""
inference 용 
24GB GPU에서는 3B 모델을 사용할 수 없습니다.
"""
class Mamba3B(nn.Module):
    def __init__(self, config):
        """
        config: Hydra에서 넘어온 DictConfig (cfg.model)
            - 예시로 config.num_labels = 3
        """
        super().__init__()


        # Mamba-3B 설정
        self.num_labels = getattr(config, "num_labels", 3)  # 기본값=3
        
        # Mamba LM config 설정
        self.mamba_config = AutoConfig.from_pretrained("state-spaces/mamba-2.8b-hf")
        self.mamba_config.output_hidden_states = True  # hidden_states 반환

        self.mamba_model = AutoModelForCausalLM.from_pretrained(
                "state-spaces/mamba-2.8b-hf",
                config = self.mamba_config,
                torch_dtype=torch.float16,  # 모델을 FP16으로 로드합니다.
                device_map="auto"  # 가용한 디바이스에 자동으로 할당합니다.
        )


        # 최종 분류 레이어
        hidden_size = self.mamba_config.hidden_size  # 일반적으로 768
        self.fc = nn.Linear(hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask):
        """
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        """
        # Mamba-3B 호출
        mamba_output = self.mamba_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 마지막 hidden state 추출
        mamba_embeddings = mamba_output.hidden_states[-1]  # (batch_size, seq_len, hidden_size)

        # 마스크 평균 풀링
        masked_output = attention_mask.unsqueeze(2) * mamba_embeddings
        mask_len = torch.sum(attention_mask, dim=1, keepdim=True)
        pooled_output = torch.sum(masked_output, dim=1) / mask_len  # (batch_size, hidden_size)

        # 최종 분류
        logits = self.fc(pooled_output)  # (batch_size, num_labels)
        return logits

# 모델 초기화 및 파라미터 수 계산
if __name__ == "__main__":
    from parameter_counter import count_parameters
    from omegaconf import DictConfig

    # 예시 Config 정의
    config = DictConfig({
        "num_labels": 3  # 분류 클래스 수 설정
    })

    # 모델 생성
    model = Mamba3B(config)

    # 파라미터 수 계산
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")  # 총 파라미터 수를 출력