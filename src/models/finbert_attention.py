import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

# 기본 구성 요소: RMSNorm 레이어
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

# 단순 Self-Attention 기반 Adapter 블록
class AttentionAdapterBlock(nn.Module):
    """
    AttentionAdapterBlock: RMSNorm -> Self-Attention -> Dropout -> Residual 연결 ->
                           RMSNorm -> MLP (Dropout 포함) -> Residual 연결
    """
    def __init__(self, d_model, num_heads=8, dropout_rate=0.1, rmsnorm_eps=1e-6):
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps=rmsnorm_eps)
        # MultiheadAttention: batch_first=True를 사용하여 입력 shape이 [B, L, d_model]임을 명시
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout_rate, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm2 = RMSNorm(d_model, eps=rmsnorm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        # x: [B, L, d_model] (여기서는 L=1, 즉 [CLS] 토큰 임베딩)
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_output)  # Residual 연결 1
        x = x + self.mlp(self.norm2(x))       # Residual 연결 2
        return x

# PEFT 방식 모델: FinBERT에 단순 AttentionAdapter 기반 adapter 및 동적 adapter fusion 적용
class FinBERT_AttentionAdapter(nn.Module):
    def __init__(self, config):
        """
        config:
          - config.embedding: FinBERT 모델 경로 (예: "ProsusAI/finbert")
          - config.num_labels: 분류 클래스 수
          - config.dropout_rate: adapter 및 내부 블록 dropout 비율
          - config.rmsnorm_eps: RMSNorm epsilon 값
          - config.adapter_layer: adapter 블록 수
          - config.adapter_heads: adapter self-attention head 수
          - config.adapter_dim: (선택사항) adapter fusion에 사용할 hidden dimension (보통 d_model과 동일)
        """
        super().__init__()
        self.num_labels = config.num_labels
        self.embedding = config.embedding

        # 사전학습된 FinBERT 모델 로드 (hidden_states 포함)
        self.config_model = AutoConfig.from_pretrained(self.embedding, output_hidden_states=True)
        self.finetuned_model = AutoModel.from_pretrained(self.embedding, config=self.config_model)

        # PEFT 방식: 사전학습 모델 파라미터 freeze
        for param in self.finetuned_model.parameters():
            param.requires_grad = False

        d_model = self.config_model.hidden_size

        # Adapter Layers: config.adapter_layer 개의 AttentionAdapterBlock을 계층적으로 쌓음
        num_layers = config.adapter_layer if hasattr(config, "adapter_layer") else 2
        num_heads = config.adapter_heads if hasattr(config, "adapter_heads") else 8
        self.adapter_layers = nn.ModuleList([
            AttentionAdapterBlock(
                d_model=d_model,
                num_heads=num_heads,
                dropout_rate=config.dropout_rate,
                rmsnorm_eps=config.rmsnorm_eps
            )
            for _ in range(num_layers)
        ])

        # Adapter fusion: 각 adapter layer의 출력을 동적으로 융합하기 위한 gating 네트워크
        # 보통 d_model과 동일한 차원으로 설정
        self.adapter_gate = nn.Linear(d_model, num_layers)
        self.gate_activation = nn.Softmax(dim=-1)

        # 최종 분류 헤드
        self.classifier = nn.Linear(d_model, self.num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.finetuned_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states[-1]  # [B, L, d_model]
        cls_output = hidden_states[:, 0, :]         # [B, d_model] -> [CLS] 토큰 임베딩

        # 각 adapter layer의 출력 계산: 입력 shape을 [B, 1, d_model]로 확장하여 adapter 적용
        adapter_outputs = []
        x = cls_output.unsqueeze(1)  # [B, 1, d_model]
        for adapter in self.adapter_layers:
            x = adapter(x)
            adapter_outputs.append(x.squeeze(1))  # 각 출력: [B, d_model]

        # 스택 후 gating fusion 수행: [B, num_layers, d_model] -> [B, d_model]
        stacked_adapters = torch.stack(adapter_outputs, dim=1)
        gate_logits = self.adapter_gate(cls_output)      # [B, num_layers]
        gate_weights = self.gate_activation(gate_logits)   # [B, num_layers]
        fused_adapter = torch.sum(stacked_adapters * gate_weights.unsqueeze(-1), dim=1)

        # Residual 연결: 원래의 [CLS] 임베딩과 fused adapter 결합 후 분류
        combined = cls_output + fused_adapter
        logits = self.classifier(combined)
        return logits
