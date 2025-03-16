import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from mamba_ssm import Mamba, Mamba2

# RMSNorm 정의
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

# MambaBlock 정의 (Dropout 적용)
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

# PEFT 방식 모델: 사전학습된 모델의 대부분은 freeze하고, adapter(MambaBlock)를 추가함.
class MambaPEFT(nn.Module):
    def __init__(self, config):
        """
        config:
          - embedding: "state-spaces/mamba-2.8b-hf"
          - num_labels: 분류 클래스 수 (예: 3)
          - dropout_rate: adapter dropout 비율 (예: 0.1)
          - rmsnorm_eps: RMSNorm epsilon (예: 1e-6)
          - mamba_block: { d_state, d_conv, expand }
        """
        super().__init__()
        self.num_labels = config.num_labels
        self.embedding = config.embedding

        self.embedding = "state-spaces/mamba-2.8b-hf"
        # 사전학습된 모델 로드 (여기서는 Mamba 2.8B)
        self.config_model = AutoConfig.from_pretrained(self.embedding, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(self.embedding, config=self.config_model)
        
        # PEFT 방식: 사전학습 모델의 파라미터는 고정
        for param in self.model.parameters():
            param.requires_grad = False

        # Adapter로 MambaBlock 사용
        d_model = self.config_model.hidden_size
        self.adapter = MambaBlock(
            d_model=d_model,
            d_state=config.mamba_block.d_state,
            d_conv=config.mamba_block.d_conv,
            expand=config.mamba_block.expand,
            dropout_rate=config.dropout_rate,
            rmsnorm_eps=config.rmsnorm_eps
        )
        
        # 최종 분류 헤드
        self.classifier = nn.Linear(d_model, self.num_labels)

    def forward(self, input_ids, attention_mask):
        # Finetuning을 위한 출력: hidden_states[-1]에서 [CLS] 토큰 추출
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states[-1]  # [B, L, d_model]
        cls_output = hidden_states[:, 0, :]         # [B, d_model]
        
        # Adapter 적용 (PEFT)
        adapter_output = self.adapter(cls_output.unsqueeze(1)).squeeze(1)
        combined = cls_output + adapter_output  # Residual 연결
        
        logits = self.classifier(combined)
        return logits
