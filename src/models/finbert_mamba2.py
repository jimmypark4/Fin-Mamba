import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from mamba_ssm import Mamba, Mamba2

# 기본 구성 요소
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
    def __init__(self, d_model, d_state, d_conv, expand, dropout_rate=0.2, rmsnorm_eps=1e-6):
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

# Enhanced PEFT Model with Multi-layer Adapter and Gating Fusion
class FinBERT_Mamba2(nn.Module):
    def __init__(self, config):
        """
        config:
          - config.embedding: FinBERT 모델 경로 (예: "ProsusAI/finbert")
          - config.num_labels: 분류 클래스 수
          - config.dropout_rate: adapter dropout 비율
          - config.rmsnorm_eps: RMSNorm epsilon 값
          - config.mamba_block: { d_state, d_conv, expand, num_adapter_layers }
          - config.adapter_gate_dim: adapter fusion에 사용할 hidden dimension (보통 d_model과 동일)
        """
        super().__init__()
        self.num_labels = config.num_labels
        self.embedding = config.embedding

        # FinBERT 사전학습 모델 로드 (hidden_states 포함)
        self.config_model = AutoConfig.from_pretrained(self.embedding, output_hidden_states=True)
        self.finetuned_model = AutoModel.from_pretrained(self.embedding, config=self.config_model)

        # PEFT 방식: 사전학습 모델 파라미터 동결
        for param in self.finetuned_model.parameters():
            param.requires_grad = False

        d_model = self.config_model.hidden_size
        num_layers = config.mamba_block.get("num_adapter_layers", 2)
        self.adapter_layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=config.mamba_block.d_state,
                d_conv=config.mamba_block.d_conv,
                expand=config.mamba_block.expand,
                dropout_rate=config.dropout_rate,
                rmsnorm_eps=config.rmsnorm_eps
            )
            for _ in range(num_layers)
        ])

        # Adapter fusion: 학습 가능한 gating으로 각 adapter 출력의 가중치를 결정
        self.adapter_gate = nn.Linear(d_model, num_layers)
        self.gate_activation = nn.Softmax(dim=-1)

        # 최종 분류 헤드
        self.classifier = nn.Linear(d_model, self.num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.finetuned_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states[-1]  # [B, L, d_model]
        cls_output = hidden_states[:, 0, :]         # [B, d_model]

        # 각 adapter layer의 출력 계산
        adapter_outputs = []
        x = cls_output.unsqueeze(1)  # [B, 1, d_model]
        for adapter in self.adapter_layers:
            x = adapter(x)
            adapter_outputs.append(x.squeeze(1))  # 각 adapter 출력: [B, d_model]

        # 스택하여 fusion 수행
        stacked_adapters = torch.stack(adapter_outputs, dim=1)  # [B, num_layers, d_model]
        gate_logits = self.adapter_gate(cls_output)             # [B, num_layers]
        gate_weights = self.gate_activation(gate_logits)         # [B, num_layers]
        fused_adapter = torch.sum(stacked_adapters * gate_weights.unsqueeze(-1), dim=1)  # [B, d_model]

        # Residual 연결: 원래 [CLS] 임베딩과 fused adapter 결합
        combined = cls_output + fused_adapter
        logits = self.classifier(combined)
        return logits
