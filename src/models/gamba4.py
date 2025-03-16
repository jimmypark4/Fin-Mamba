import torch
import torch.nn as nn
from mamba_ssm import Mamba
from mamba_ssm import Mamba2
from transformers import AutoModelForCausalLM, AutoConfig

# RMSNorm 레이어 (변경 없음)
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = torch.sqrt(norm.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x / rms

# MambaBlock: RMSNorm -> Mamba2 -> RMSNorm -> MLP
class MambaBlock(nn.Module):
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

# AttentionBlock: RMSNorm -> MultiHeadAttention -> RMSNorm -> MLP
class AttentionBlock(nn.Module):
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

# BaseSubModel: 서브모델들의 공통 인터페이스 정의
class BaseSubModel(nn.Module):
    def forward(self, embeddings, attention_mask):
        raise NotImplementedError("서브클래스에서 구현하세요.")

# SubModel1: 기본 (6 MambaBlock : 1 AttentionBlock) 구성
class SubModel1(BaseSubModel):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList()
        d_model   = config.mamba_block.d_model
        d_state   = config.mamba_block.d_state
        d_conv    = config.mamba_block.d_conv
        expand_mb = config.mamba_block.expand
        n_heads   = config.attention_block.n_heads
        expand_attn = config.attention_block.expand
        eps       = config.rmsnorm_eps
        num_cycles= config.num_cycles

        for _ in range(num_cycles):
            for _ in range(6):
                self.blocks.append(
                    MambaBlock(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand_mb,
                        rmsnorm_eps=eps
                    )
                )
            self.blocks.append(
                AttentionBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    expand=expand_attn,
                    rmsnorm_eps=eps
                )
            )

    def forward(self, embeddings, attention_mask):
        # 1) 마스크를 적용한 평균 풀링
        masked = attention_mask.unsqueeze(2) * embeddings  # [B, L, H]
        mask_len = torch.sum(attention_mask, dim=1, keepdim=True)  # [B, 1]
        pooled = torch.sum(masked, dim=1) / (mask_len + 1e-8)  # [B, H]
        # 2) [B, 1, H]로 변환 후 블록 적용
        x = pooled.unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        # 3) 다시 [B, H]로 변환하여 반환
        return x.squeeze(1)

# SubModel2: 기본 구성 (필요에 따라 나중에 다른 아키텍쳐로 변경 가능)
class SubModel2(BaseSubModel):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList()
        d_model   = config.mamba_block.d_model
        d_state   = config.mamba_block.d_state
        d_conv    = config.mamba_block.d_conv
        expand_mb = config.mamba_block.expand
        n_heads   = config.attention_block.n_heads
        expand_attn = config.attention_block.expand
        eps       = config.rmsnorm_eps
        num_cycles= config.num_cycles

        for _ in range(num_cycles):
            for _ in range(6):
                self.blocks.append(
                    MambaBlock(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand_mb,
                        rmsnorm_eps=eps
                    )
                )
            self.blocks.append(
                AttentionBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    expand=expand_attn,
                    rmsnorm_eps=eps
                )
            )

    def forward(self, embeddings, attention_mask):
        masked = attention_mask.unsqueeze(2) * embeddings
        mask_len = torch.sum(attention_mask, dim=1, keepdim=True)
        pooled = torch.sum(masked, dim=1) / (mask_len + 1e-8)
        x = pooled.unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        return x.squeeze(1)

# SubModel3: 기본 구성 (필요에 따라 나중에 다른 아키텍쳐로 변경 가능)
class SubModel3(BaseSubModel):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList()
        d_model   = config.mamba_block.d_model
        d_state   = config.mamba_block.d_state
        d_conv    = config.mamba_block.d_conv
        expand_mb = config.mamba_block.expand
        n_heads   = config.attention_block.n_heads
        expand_attn = config.attention_block.expand
        eps       = config.rmsnorm_eps
        num_cycles= config.num_cycles

        for _ in range(num_cycles):
            for _ in range(6):
                self.blocks.append(
                    MambaBlock(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand_mb,
                        rmsnorm_eps=eps
                    )
                )
            self.blocks.append(
                AttentionBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    expand=expand_attn,
                    rmsnorm_eps=eps
                )
            )

    def forward(self, embeddings, attention_mask):
        masked = attention_mask.unsqueeze(2) * embeddings
        mask_len = torch.sum(attention_mask, dim=1, keepdim=True)
        pooled = torch.sum(masked, dim=1) / (mask_len + 1e-8)
        x = pooled.unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        return x.squeeze(1)

# Gamba4: Mamba LM -> 서브모델1, 2, 3 병렬 -> 앙상블 -> 최종 분류
class Gamba4(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels

        # 1) 사전 학습된 Mamba LM 설정
        embedding = "state-spaces/mamba-130m-hf"
        self.mamba_config = AutoConfig.from_pretrained(embedding)
        self.mamba_config.output_hidden_states = True
        self.mamba_model = AutoModelForCausalLM.from_pretrained(embedding, config=self.mamba_config)
        mamba_hidden_size = self.mamba_config.hidden_size

        # Mamba LM의 hidden size를 서브모델과 맞춤
        config.mamba_block.d_model = mamba_hidden_size

        # 2) 서브모델 1, 2, 3을 각각 생성 (나중에 개별 아키텍처로 교체 가능)
        self.sub_model1 = SubModel1(config)
        self.sub_model2 = SubModel2(config)
        self.sub_model3 = SubModel3(config)

        # 3) 최종 분류 레이어
        self.fc = nn.Linear(mamba_hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask):
        # A. Mamba LM으로부터 임베딩 추출
        mamba_output = self.mamba_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        embeddings = mamba_output.hidden_states[-1]  # [B, L, H]

        # B. 각 서브모델을 개별로 처리
        out1 = self.sub_model1(embeddings, attention_mask)
        out2 = self.sub_model2(embeddings, attention_mask)
        out3 = self.sub_model3(embeddings, attention_mask)

        # C. 서브모델들의 결과를 앙상블 (예: 평균)
        ensemble_output = (out1 + out2 + out3) / 3
        # D. 최종 분류
        logits = self.fc(ensemble_output)
        return logits
