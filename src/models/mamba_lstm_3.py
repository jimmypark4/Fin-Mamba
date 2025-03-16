import torch  # 텐서 연산을 위한 PyTorch 임포트
import torch.nn as nn  # 신경망 모델을 위한 nn 모듈 임포트
from mamba_ssm import Mamba  # mamba_ssm 라이브러리에서 Mamba 모델 임포트
from transformers import AutoModelForCausalLM, AutoConfig  # 사전 학습된 언어 모델 및 설정 임포트

class RMSNorm(nn.Module):
    """RMSNorm 레이어"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))  # RMSNorm의 가중치 파라미터 초기화
        self.eps = eps  # 수치적 안정성을 위한 epsilon 값

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)  # 입력 텐서의 L2 노름 계산
        rms = torch.sqrt(norm.pow(2).mean(-1, keepdim=True) + self.eps)  # 평균 제곱근 노름 계산
        return self.weight * x / rms  # 입력 텐서를 정규화하고 학습된 가중치로 스케일링

class MambaBlock(nn.Module):
    """
    MambaBlock: RMSNorm -> Mamba -> RMSNorm -> MLP
    """
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.norm1 = RMSNorm(d_model)  # 첫 번째 RMSNorm 레이어
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)  # Mamba 레이어
        self.norm2 = RMSNorm(d_model)  # 두 번째 RMSNorm 레이어
        self.mlp = nn.Sequential(  # 다층 퍼셉트론(MLP) 레이어
            nn.Linear(d_model, d_model * 4),  # 모델 크기를 4배 확장하는 선형층
            nn.GELU(),  # GELU 활성화 함수
            nn.Linear(d_model * 4, d_model)  # 다시 원래 크기로 축소하는 선형층
        )

    def forward(self, x):
        # RMSNorm -> Mamba를 통한 잔차 연결
        x = x + self.mamba(self.norm1(x))
        # RMSNorm -> MLP를 통한 잔차 연결
        x = x + self.mlp(self.norm2(x))
        return x  # 처리된 출력 반환

class MambaLstm_3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = getattr(config, "num_labels", 3)  # 라벨 개수 설정

        # Mamba LM 설정
        self.mamba_config = AutoConfig.from_pretrained("state-spaces/mamba-370m-hf")  # 사전 학습된 Mamba 모델 설정 로드
        self.mamba_config.output_hidden_states = True  # 은닉 상태 출력 설정
        self.mamba_model = AutoModelForCausalLM.from_pretrained(
            "state-spaces/mamba-370m-hf",  # 사전 학습된 모델 로드
            config=self.mamba_config
        )

        mamba_hidden_size = self.mamba_config.hidden_size  # ex) 1024

        # Bi-LSTM 설정
        rnn = nn.LSTM  # LSTM 레이어 정의
        lstm_hidden_size = 200  # LSTM의 은닉층 크기 설정

         # rnn1, rnn2의 input_size를 모델의 hidden_size로 맞춤
        self.rnn1 = rnn(mamba_hidden_size, lstm_hidden_size, bidirectional=True)
        self.rnn2 = rnn(lstm_hidden_size, lstm_hidden_size, bidirectional=True)

        # MambaBlock들 (7개 레이어)
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(d_model=800, d_state=16, d_conv=4, expand=2) for _ in range(11)  # MambaBlock을 11개 쌓기
        ])

        # 최종 분류 레이어
        self.fc = nn.Linear(800, self.num_labels)  # 800 차원의 출력을 라벨 수로 변환

    def forward(self, input_ids, attention_mask):
        # A. Mamba-130M 모델 호출
        mamba_output = self.mamba_model(input_ids=input_ids, attention_mask=attention_mask)  # Mamba 모델로부터 출력 받기
        mamba_embeddings = mamba_output.hidden_states[-1]  # 마지막 은닉 상태 임베딩 추출

        # B. Bi-LSTM 처리
        masked_output = attention_mask.unsqueeze(2) * mamba_embeddings  # 마스크를 고려한 출력 계산
        mask_len = torch.sum(attention_mask, dim=1, keepdim=True)  # 마스크된 길이 계산
        mamba_pooled = torch.sum(masked_output, dim=1) / mask_len  # 평균 풀링

        utterance_text = mamba_pooled.unsqueeze(0)  # 차원 추가
        _, (h1, _) = self.rnn1(utterance_text)  # 첫 번째 Bi-LSTM 처리
        _, (h2, _) = self.rnn2(h1)  # 두 번째 Bi-LSTM 처리
        lstm_output = torch.cat([h1, h2], dim=2)  # 두 LSTM의 출력을 합치기
        lstm_output = lstm_output.permute(1, 0, 2).contiguous().view(h1.shape[1], -1)  # 차원 변환

        # C. MambaBlock 처리 (7개의 블록을 순차적으로 처리)
        lstm_output = lstm_output.unsqueeze(1)  # 차원 추가 (배치 크기, 1, 800)
        for mamba_block in self.mamba_blocks:
            lstm_output = mamba_block(lstm_output)  # 각 MambaBlock을 통과
        lstm_output = lstm_output.squeeze(1)  # 차원 축소 (배치 크기, 800)

        # D. 최종 분류
        logits = self.fc(lstm_output)  # 최종 분류 출력 계산
        return logits  # 로짓 반환

# 모델 초기화 및 파라미터 수 계산
if __name__ == "__main__":
    from parameter_counter import count_parameters  # 파라미터 수 계산 함수 임포트
    from omegaconf import DictConfig  # 설정 파일을 위한 DictConfig 임포트

    # 예시 Config 정의
    config = DictConfig({
        "num_labels": 3  # 라벨 수 설정
    })

    # 모델 생성
    model = MambaLstm_3(config)

    # 파라미터 수 계산
    total_params = count_parameters(model)  # 모델의 파라미터 수 계산
    print(f"Total parameters: {total_params:,}")  # 파라미터 수 출력
