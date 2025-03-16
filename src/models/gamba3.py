import torch
import torch.nn as nn
from mamba_ssm import Mamba
from mamba_ssm import Mamba2

from transformers import AutoModelForCausalLM, AutoConfig

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
    MambaBlock: RMSNorm -> Mamba -> RMSNorm -> MLP
    """
    def __init__(self, d_model, d_state, d_conv, expand, rmsnorm_eps=1e-6):
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps=rmsnorm_eps)
        self.mamba = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm2 = RMSNorm(d_model, eps=rmsnorm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        x = x + self.mamba(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class AttentionBlock(nn.Module):
    """
    AttentionBlock: RMSNorm -> MultiHeadAttention -> RMSNorm -> MLP
    """
    def __init__(self, d_model, n_heads=8, expand=4, rmsnorm_eps=1e-6):
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps=rmsnorm_eps)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm2 = RMSNorm(d_model, eps=rmsnorm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * expand),
            nn.GELU(),
            nn.Linear(d_model * expand, d_model)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output
        x_norm2 = self.norm2(x)
        mlp_out = self.mlp(x_norm2)
        x = x + mlp_out
        return x

class Gamba3(nn.Module):
    """
    Gamba3: 사전 학습된 Mamba LM + (6 MambaBlock : 1 AttentionBlock) 사이클을 통해
           특징을 추출하고 분류하는 모델.
    """
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        
        # 사전 학습된 Mamba LM 설정 (embedding 경로는 config에서)
        embedding = "state-spaces/mamba-130m-hf"
        self.mamba_config = AutoConfig.from_pretrained(embedding)
        self.mamba_config.output_hidden_states = True
        self.mamba_model = AutoModelForCausalLM.from_pretrained(embedding, config=self.mamba_config)
        mamba_hidden_size = self.mamba_config.hidden_size

        # (6 MambaBlock : 1 AttentionBlock) 사이클 반복
        num_cycles = config.num_cycles
        self.blocks = nn.ModuleList()

        for _ in range(num_cycles):
            for _ in range(6):
                self.blocks.append(MambaBlock(
                    d_model=mamba_hidden_size,
                    d_state=config.mamba_block.d_state,
                    d_conv=config.mamba_block.d_conv,
                    expand=config.mamba_block.expand,
                    rmsnorm_eps=config.rmsnorm_eps
                ))
            self.blocks.append(AttentionBlock(
                d_model=mamba_hidden_size,
                n_heads=config.attention_block.n_heads,
                expand=config.attention_block.expand,
                rmsnorm_eps=config.rmsnorm_eps
            ))

        # 최종 분류 레이어
        self.fc = nn.Linear(mamba_hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask):
        # A. Mamba LM 임베딩 추출
        mamba_output = self.mamba_model(input_ids=input_ids, attention_mask=attention_mask)
        mamba_embeddings = mamba_output.hidden_states[-1]

        # B. 마스크 적용 및 평균 풀링
        masked_output = attention_mask.unsqueeze(2) * mamba_embeddings
        mask_len = torch.sum(attention_mask, dim=1, keepdim=True)
        pooled_output = torch.sum(masked_output, dim=1) / mask_len
        pooled_output = pooled_output.unsqueeze(1)

        # C. 블록 순차 처리
        for block in self.blocks:
            pooled_output = block(pooled_output)
        pooled_output = pooled_output.squeeze(1)

        # D. 최종 분류
        logits = self.fc(pooled_output)
        return logits
