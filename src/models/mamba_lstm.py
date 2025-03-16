import torch
import torch.nn as nn
from mamba_ssm import Mamba
from transformers import AutoModelForCausalLM, AutoConfig

class LstmMamba_2(nn.Module):
    def __init__(self, config):
        """
        config: Hydra에서 넘어온 DictConfig (cfg.model)
            - 예시로 config.num_labels = 2 또는 3
        """
        super().__init__()
        # 1) Mamba-130M 설정
        self.num_labels = getattr(config, "num_labels", 3)  # 기본값=3
        
        # Mamba LM config
        self.mamba_config = AutoConfig.from_pretrained("state-spaces/mamba-130m-hf")
        # hidden_states를 반환하도록 세팅
        self.mamba_config.output_hidden_states = True

        self.mamba_model = AutoModelForCausalLM.from_pretrained(
            "state-spaces/mamba-130m-hf",
            config=self.mamba_config
        )

        # 2) Bi-LSTM 준비
        rnn = nn.LSTM
        lstm_hidden_size = 200
        # mamba hidden_dim = 768 (가정)
        self.rnn1 = rnn(768, lstm_hidden_size, bidirectional=True)
        self.rnn2 = rnn(lstm_hidden_size, lstm_hidden_size, bidirectional=True)

        # 3) Mamba 블록
        self.mamba1 = Mamba(d_model=800, d_state=16, d_conv=4, expand=2)
        self.mamba2 = Mamba(d_model=800, d_state=16, d_conv=4, expand=2)

        # 4) 최종 분류 레이어 (num_labels만큼 출력)
        self.fc = nn.Linear(800, self.num_labels)

    def forward(self, input_ids, attention_mask):
        """
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        """
        # A. Mamba-130M 모델 호출 (임베딩 생성)
        mamba_output = self.mamba_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # 마지막 레이어 임베딩
        mamba_embeddings = mamba_output.hidden_states[-1]  # (batch_size, seq_len, 130)

        # B. Bi-LSTM
        # 마스크 평균 풀링
        masked_output = attention_mask.unsqueeze(2) * mamba_embeddings
        mask_len = torch.sum(attention_mask, dim=1, keepdim=True)
        mamba_pooled = torch.sum(masked_output, dim=1) / mask_len  # (batch_size, 130)

        # LSTM 입력 준비
        utterance_text = mamba_pooled.unsqueeze(0)  # (1, batch_size, 130)
        _, (h1, _) = self.rnn1(utterance_text)
        _, (h2, _) = self.rnn2(h1)
        # h1, h2 shape: (2, batch_size, 200)

        # LSTM 출력 결합
        lstm_output = torch.cat([h1, h2], dim=2)  # (2, batch_size, 400)
        lstm_output = lstm_output.permute(1, 0, 2).contiguous().view(h1.shape[1], -1)  # (batch_size, 800)

        # C. Mamba 블록 처리
        lstm_output = lstm_output.unsqueeze(1)  # (batch_size, 1, 800)
        mamba_out1 = self.mamba1(lstm_output)
        mamba_out2 = self.mamba2(mamba_out1).squeeze(1)  # (batch_size, 800)

        # D. 최종 분류
        logits = self.fc(mamba_out2)  # (batch_size, num_labels)
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
    model = LstmMamba_2(config)

    # 파라미터 수 계산
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")  # 총 파라미터 수를 출력