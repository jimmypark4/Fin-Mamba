import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba, Mamba2
from transformers import AutoModelForCausalLM, AutoConfig

#############################################
# 공통 레이어 및 블록 정의 (Dropout 적용)
#############################################

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

class AttentionBlock(nn.Module):
    """
    AttentionBlock: RMSNorm -> MultiHeadAttention -> RMSNorm -> MLP (Dropout 적용)
    """
    def __init__(self, d_model, n_heads=8, expand=4, dropout_rate=0.1, rmsnorm_eps=1e-6):
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps=rmsnorm_eps)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm2 = RMSNorm(d_model, eps=rmsnorm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * expand),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * expand, d_model),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_output)
        x_norm2 = self.norm2(x)
        mlp_out = self.mlp(x_norm2)
        x = x + mlp_out
        return x

#############################################
# 서브모델 예시: 3개의 블록만 (MambaBlock 혹은 AttentionBlock)
#############################################

class BaseSubModel(nn.Module):
    def forward(self, embeddings, attention_mask):
        raise NotImplementedError("Subclass must implement forward()")

class SubModel1(BaseSubModel):
    """ 3개의 MambaBlock """
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=config.mamba_block.d_model,
                d_state=config.mamba_block.d_state,
                d_conv=config.mamba_block.d_conv,
                expand=config.mamba_block.expand,
                dropout_rate=config.dropout_rate,
                rmsnorm_eps=config.rmsnorm_eps
            )
            for _ in range(3)
        ])

    def forward(self, embeddings, attention_mask):
        # 평균 풀링
        masked = embeddings * attention_mask.unsqueeze(2)  # [B, L, H]
        lengths = torch.sum(attention_mask, dim=1, keepdim=True) + 1e-8
        pooled = torch.sum(masked, dim=1) / lengths  # [B, H]
        x = pooled.unsqueeze(1)                      # [B, 1, H]
        for blk in self.blocks:
            x = blk(x)
        return x.squeeze(1)                         # [B, H]

class SubModel2(BaseSubModel):
    """ 3개의 AttentionBlock """
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([
            AttentionBlock(
                d_model=config.mamba_block.d_model,
                n_heads=config.attention_block.n_heads,
                expand=config.attention_block.expand,
                dropout_rate=config.dropout_rate,
                rmsnorm_eps=config.rmsnorm_eps
            )
            for _ in range(3)
        ])

    def forward(self, embeddings, attention_mask):
        masked = embeddings * attention_mask.unsqueeze(2)
        lengths = torch.sum(attention_mask, dim=1, keepdim=True) + 1e-8
        pooled = torch.sum(masked, dim=1) / lengths
        x = pooled.unsqueeze(1)
        for blk in self.blocks:
            x = blk(x)
        return x.squeeze(1)

class SubModel3(BaseSubModel):
    """ 혼합형: MambaBlock -> AttentionBlock -> MambaBlock """
    def __init__(self, config):
        super().__init__()
        self.block1 = MambaBlock(
            d_model=config.mamba_block.d_model,
            d_state=config.mamba_block.d_state,
            d_conv=config.mamba_block.d_conv,
            expand=config.mamba_block.expand,
            dropout_rate=config.dropout_rate,
            rmsnorm_eps=config.rmsnorm_eps
        )
        self.block2 = AttentionBlock(
            d_model=config.mamba_block.d_model,
            n_heads=config.attention_block.n_heads,
            expand=config.attention_block.expand,
            dropout_rate=config.dropout_rate,
            rmsnorm_eps=config.rmsnorm_eps
        )
        self.block3 = MambaBlock(
            d_model=config.mamba_block.d_model,
            d_state=config.mamba_block.d_state,
            d_conv=config.mamba_block.d_conv,
            expand=config.mamba_block.expand,
            dropout_rate=config.dropout_rate,
            rmsnorm_eps=config.rmsnorm_eps
        )

    def forward(self, embeddings, attention_mask):
        masked = embeddings * attention_mask.unsqueeze(2)
        lengths = torch.sum(attention_mask, dim=1, keepdim=True) + 1e-8
        pooled = torch.sum(masked, dim=1) / lengths
        x = pooled.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x.squeeze(1)

#############################################
# 간단한 Softmax Gating (Expert 앙상블)
#############################################

class SimpleSoftmaxGating(nn.Module):
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.linear = nn.Linear(d_model, num_experts)

    def forward(self, pooled_rep):
        # softmax로 3개 서브모델에 대한 가중치 할당
        logits = self.linear(pooled_rep)  # [B, num_experts]
        weights = F.softmax(logits, dim=-1)  # [B, num_experts]
        return weights

#############################################
# Gamba7: 3개 서브모델 + Softmax Gating -> 최종 분류
#############################################

class Gamba7(nn.Module):
    """
    구조:
    - Mamba LM에서 임베딩 추출
    - 3개의 서브모델(간소화된 구조)을 통한 특징 추출
    - Softmax 기반 게이팅 네트워크로 가중 합산
    - 최종 분류 레이어
    """
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels

        # 사전학습된 Mamba LM 설정
        embedding = config.embedding
        self.mamba_config = AutoConfig.from_pretrained(embedding)
        self.mamba_config.output_hidden_states = True
        self.mamba_model = AutoModelForCausalLM.from_pretrained(embedding, config=self.mamba_config)
        mamba_hidden_size = self.mamba_config.hidden_size

        # Mamba LM의 hidden size를 서브모델과 맞춤
        config.mamba_block.d_model = mamba_hidden_size

        # 서브모델 3개로 축소
        self.sub_model1 = SubModel1(config)
        self.sub_model2 = SubModel2(config)
        self.sub_model3 = SubModel3(config)

        # 게이팅 (Softmax 기반)
        self.gating = SimpleSoftmaxGating(d_model=mamba_hidden_size, num_experts=3)

        # 최종 분류 레이어
        self.fc = nn.Linear(mamba_hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask):
        # (A) 임베딩 추출
        mamba_output = self.mamba_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = mamba_output.hidden_states[-1]  # [B, L, H]

        # (B) 서브모델 처리
        out1 = self.sub_model1(embeddings, attention_mask)
        out2 = self.sub_model2(embeddings, attention_mask)
        out3 = self.sub_model3(embeddings, attention_mask)

        # (C) 메인 Pooled Representation (게이팅에 사용)
        masked = embeddings * attention_mask.unsqueeze(2)
        lengths = torch.sum(attention_mask, dim=1, keepdim=True)
        pooled_rep = torch.sum(masked, dim=1) / (lengths + 1e-8)  # [B, H]

        # (D) Softmax Gating
        gate_weights = self.gating(pooled_rep)  # [B, 3]

        # (E) 가중 합산
        outputs = torch.stack([out1, out2, out3], dim=1)   # [B, 3, H]
        ensemble_output = torch.sum(outputs * gate_weights.unsqueeze(-1), dim=1)  # [B, H]

        # (F) 최종 분류
        logits = self.fc(ensemble_output)
        return logits
