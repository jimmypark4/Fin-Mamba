import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from mamba_ssm import Mamba, Mamba2

# 기존에 사용한 RMSNorm, MambaBlock 정의 (Dropout 포함)
class RMSNorm(nn.Module):
    """RMSNorm 레이어"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = torch.sqrt(norm.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x / rms

class MambaBlock(nn.Module):
    """
    MambaBlock: RMSNorm -> Mamba2 -> RMSNorm -> MLP (Dropout 적용)
    """
    def __init__(self, d_model, d_state, d_conv, expand, dropout_rate=0.1, rmsnorm_eps=1e-6):
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps=rmsnorm_eps)
        self.mamba = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm2 = RMSNorm(d_model, eps=rmsnorm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        x = x + self.dropout(self.mamba(self.norm1(x)))
        x = x + self.mlp(self.norm2(x))
        return x

# PEFT 방식 FinBERT + MambaBlock adapter
class FinBERT_MambaAdapter(nn.Module):
    def __init__(self, config):
        """
        config: Dict-like 객체로,
          - config.embedding: FinBERT 모델 경로 (예: "ProsusAI/finbert")
          - config.num_labels: 분류 클래스 수
          - config.dropout_rate: adapter 및 내부 블록 dropout 비율
          - config.rmsnorm_eps: RMSNorm epsilon 값
          - config.mamba_block: { d_state, d_conv, expand } 등의 adapter 관련 파라미터
        """
        super().__init__()
        self.num_labels = config.num_labels
        self.embedding = config.embedding

        # FinBERT 사전학습 모델 로드 (출력으로 hidden_states 포함)
        self.config_model = AutoConfig.from_pretrained(self.embedding, output_hidden_states=True)
        self.finetuned_model = AutoModel.from_pretrained(self.embedding, config=self.config_model)

        # PEFT 방식: 사전학습 모델의 파라미터를 freeze
        for param in self.finetuned_model.parameters():
            param.requires_grad = False

        # Adapter: MambaBlock을 adapter로 사용하여 추가 파라미터만 학습
        d_model = self.config_model.hidden_size
        self.adapter = MambaBlock(
            d_model=d_model,
            d_state=config.mamba_block.d_state,
            d_conv=config.mamba_block.d_conv,
            expand=config.mamba_block.expand,
            dropout_rate=config.dropout_rate,
            rmsnorm_eps=config.rmsnorm_eps
        )
        # 분류 헤드: adapter의 출력과 원래의 [CLS] 토큰을 조합하여 분류
        self.classifier = nn.Linear(d_model, self.num_labels)

    def forward(self, input_ids, attention_mask):
        # FinBERT를 통해 hidden_states 추출 (예: 마지막 레이어)
        outputs = self.finetuned_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states[-1]  # [B, L, d_model]
        # [CLS] 토큰의 임베딩 사용 (일반적으로 첫 번째 토큰)
        cls_output = hidden_states[:, 0, :]  # [B, d_model]

        # Adapter를 통해 미세조정 가능한 부분만 업데이트 (PEFT)
        # 입력을 [B, 1, d_model] 형태로 맞춘 후 adapter 적용
        adapter_input = cls_output.unsqueeze(1)  # [B, 1, d_model]
        adapter_output = self.adapter(adapter_input).squeeze(1)  # [B, d_model]

        # Residual 연결: 원래 [CLS] 토큰 임베딩에 adapter 출력을 더함
        combined = cls_output + adapter_output

        # 분류 헤드
        logits = self.classifier(combined)
        return logits
