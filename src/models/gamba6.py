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

# BaseSubModel: 서브모델들의 공통 인터페이스
class BaseSubModel(nn.Module):
    def forward(self, embeddings, attention_mask):
        raise NotImplementedError("서브클래스에서 구현하세요.")

#############################################
# 8개의 서브모델 (MambaBlock, AttentionBlock 또는 혼합)
#############################################

# SubModel1: 순수 MambaBlock 기반
class SubModel1(BaseSubModel):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(config.mamba_block.num_cycles):
            for _ in range(6):
                self.blocks.append(
                    MambaBlock(
                        d_model=config.mamba_block.d_model,
                        d_state=config.mamba_block.d_state,
                        d_conv=config.mamba_block.d_conv,
                        expand=config.mamba_block.expand,
                        dropout_rate=config.dropout_rate,
                        rmsnorm_eps=config.rmsnorm_eps
                    )
                )
    def forward(self, embeddings, attention_mask):
        masked = attention_mask.unsqueeze(2) * embeddings  # [B, L, H]
        mask_len = torch.sum(attention_mask, dim=1, keepdim=True)  # [B, 1]
        pooled = torch.sum(masked, dim=1) / (mask_len + 1e-8)       # [B, H]
        x = pooled.unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        return x.squeeze(1)

# SubModel2: 순수 MambaBlock 기반 (구조 약간 확장)
class SubModel2(BaseSubModel):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList()
        cycles = config.mamba_block.num_cycles + 1
        for _ in range(cycles):
            for _ in range(6):
                self.blocks.append(
                    MambaBlock(
                        d_model=config.mamba_block.d_model,
                        d_state=config.mamba_block.d_state,
                        d_conv=config.mamba_block.d_conv,
                        expand=config.mamba_block.expand,
                        dropout_rate=config.dropout_rate,
                        rmsnorm_eps=config.rmsnorm_eps
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

# SubModel3: 순수 AttentionBlock 기반
class SubModel3(BaseSubModel):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList()
        num_layers = getattr(config.attention_block, "num_layers", 3)
        for _ in range(num_layers):
            self.blocks.append(
                AttentionBlock(
                    d_model=config.mamba_block.d_model,
                    n_heads=config.attention_block.n_heads,
                    expand=config.attention_block.expand,
                    dropout_rate=config.dropout_rate,
                    rmsnorm_eps=config.rmsnorm_eps
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

# SubModel4: 순수 AttentionBlock 기반 (층 수 다르게)
class SubModel4(BaseSubModel):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList()
        num_layers = getattr(config.attention_block, "num_layers", 3) + 1
        for _ in range(num_layers):
            self.blocks.append(
                AttentionBlock(
                    d_model=config.mamba_block.d_model,
                    n_heads=config.attention_block.n_heads,
                    expand=config.attention_block.expand,
                    dropout_rate=config.dropout_rate,
                    rmsnorm_eps=config.rmsnorm_eps
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

# SubModel5: 혼합형 (MambaBlock 후 AttentionBlock)
class SubModel5(BaseSubModel):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(config.mamba_block.num_cycles):
            for _ in range(3):
                self.blocks.append(
                    MambaBlock(
                        d_model=config.mamba_block.d_model,
                        d_state=config.mamba_block.d_state,
                        d_conv=config.mamba_block.d_conv,
                        expand=config.mamba_block.expand,
                        dropout_rate=config.dropout_rate,
                        rmsnorm_eps=config.rmsnorm_eps
                    )
                )
            self.blocks.append(
                AttentionBlock(
                    d_model=config.mamba_block.d_model,
                    n_heads=config.attention_block.n_heads,
                    expand=config.attention_block.expand,
                    dropout_rate=config.dropout_rate,
                    rmsnorm_eps=config.rmsnorm_eps
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

# SubModel6: 혼합형 (AttentionBlock 후 MambaBlock)
class SubModel6(BaseSubModel):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(config.mamba_block.num_cycles):
            self.blocks.append(
                AttentionBlock(
                    d_model=config.mamba_block.d_model,
                    n_heads=config.attention_block.n_heads,
                    expand=config.attention_block.expand,
                    dropout_rate=config.dropout_rate,
                    rmsnorm_eps=config.rmsnorm_eps
                )
            )
            for _ in range(3):
                self.blocks.append(
                    MambaBlock(
                        d_model=config.mamba_block.d_model,
                        d_state=config.mamba_block.d_state,
                        d_conv=config.mamba_block.d_conv,
                        expand=config.mamba_block.expand,
                        dropout_rate=config.dropout_rate,
                        rmsnorm_eps=config.rmsnorm_eps
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

# SubModel7: 혼합형 (MambaBlock와 AttentionBlock 번갈아 배치)
class SubModel7(BaseSubModel):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(8):
            if i % 2 == 0:
                self.blocks.append(
                    MambaBlock(
                        d_model=config.mamba_block.d_model,
                        d_state=config.mamba_block.d_state,
                        d_conv=config.mamba_block.d_conv,
                        expand=config.mamba_block.expand,
                        dropout_rate=config.dropout_rate,
                        rmsnorm_eps=config.rmsnorm_eps
                    )
                )
            else:
                self.blocks.append(
                    AttentionBlock(
                        d_model=config.mamba_block.d_model,
                        n_heads=config.attention_block.n_heads,
                        expand=config.attention_block.expand,
                        dropout_rate=config.dropout_rate,
                        rmsnorm_eps=config.rmsnorm_eps
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

# SubModel8: 혼합형 (MambaBlock 그룹 후 AttentionBlock 그룹)
class SubModel8(BaseSubModel):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(2):
            self.blocks.append(
                MambaBlock(
                    d_model=config.mamba_block.d_model,
                    d_state=config.mamba_block.d_state,
                    d_conv=config.mamba_block.d_conv,
                    expand=config.mamba_block.expand,
                    dropout_rate=config.dropout_rate,
                    rmsnorm_eps=config.rmsnorm_eps
                )
            )
        for _ in range(2):
            self.blocks.append(
                AttentionBlock(
                    d_model=config.mamba_block.d_model,
                    n_heads=config.attention_block.n_heads,
                    expand=config.attention_block.expand,
                    dropout_rate=config.dropout_rate,
                    rmsnorm_eps=config.rmsnorm_eps
                )
            )
        for _ in range(2):
            self.blocks.append(
                MambaBlock(
                    d_model=config.mamba_block.d_model,
                    d_state=config.mamba_block.d_state,
                    d_conv=config.mamba_block.d_conv,
                    expand=config.mamba_block.expand,
                    dropout_rate=config.dropout_rate,
                    rmsnorm_eps=config.rmsnorm_eps
                )
            )
        for _ in range(2):
            self.blocks.append(
                AttentionBlock(
                    d_model=config.mamba_block.d_model,
                    n_heads=config.attention_block.n_heads,
                    expand=config.attention_block.expand,
                    dropout_rate=config.dropout_rate,
                    rmsnorm_eps=config.rmsnorm_eps
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

#############################################
# GatingNetwork: Gumbel-Softmax 기반 게이팅 네트워크
#############################################

class GatingNetwork(nn.Module):
    def __init__(self, d_model, num_experts, init_temp=1.0):
        super().__init__()
        self.linear = nn.Linear(d_model, num_experts)
        self.temp = init_temp

    def forward(self, pooled_rep):
        logits = self.linear(pooled_rep)  # [B, num_experts]
        weights = F.gumbel_softmax(logits, tau=self.temp, hard=True, dim=-1)
        return weights

#############################################
# Gamba6: Mamba LM -> 8 서브모델 병렬 -> 게이팅 네트워크를 통한 앙상블 -> 최종 분류
#############################################

class Gamba6(nn.Module):
    """
    구조:
    - Mamba LM에서 임베딩 추출
    - 8개의 서브모델(전문가: 순수 MambaBlock, 순수 AttentionBlock, 혼합형)을 통한 특징 추출
    - Gumbel-Softmax 기반 게이팅 네트워크로 각 전문가의 가중치를 산출하여 조건부 결합
    - 최종 분류 레이어
    """
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels

        # 1) 사전학습된 Mamba LM 설정
        embedding = config.embedding  # config에서 embedding 경로로 제어
        self.mamba_config = AutoConfig.from_pretrained(embedding)
        self.mamba_config.output_hidden_states = True
        self.mamba_model = AutoModelForCausalLM.from_pretrained(embedding, config=self.mamba_config)
        mamba_hidden_size = self.mamba_config.hidden_size

        # Mamba LM의 hidden size를 서브모델과 맞춤
        config.mamba_block.d_model = mamba_hidden_size

        # 2) 8개의 서브모델 생성
        self.sub_model1 = SubModel1(config)
        self.sub_model2 = SubModel2(config)
        self.sub_model3 = SubModel3(config)
        self.sub_model4 = SubModel4(config)
        self.sub_model5 = SubModel5(config)
        self.sub_model6 = SubModel6(config)
        self.sub_model7 = SubModel7(config)
        self.sub_model8 = SubModel8(config)

        # 3) 게이팅 네트워크 (전문가 수: 8)
        self.gating = GatingNetwork(d_model=mamba_hidden_size, num_experts=8, init_temp=1.0)

        # 4) 최종 분류 레이어
        self.fc = nn.Linear(mamba_hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask):
        # A. Mamba LM에서 임베딩 추출 (hidden_states[-1]: [B, L, H])
        mamba_output = self.mamba_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        embeddings = mamba_output.hidden_states[-1]

        # B. 8 서브모델을 통한 특징 추출
        out1 = self.sub_model1(embeddings, attention_mask)
        out2 = self.sub_model2(embeddings, attention_mask)
        out3 = self.sub_model3(embeddings, attention_mask)
        out4 = self.sub_model4(embeddings, attention_mask)
        out5 = self.sub_model5(embeddings, attention_mask)
        out6 = self.sub_model6(embeddings, attention_mask)
        out7 = self.sub_model7(embeddings, attention_mask)
        out8 = self.sub_model8(embeddings, attention_mask)

        # C. 게이팅 네트워크를 위한 pooled representation (서브모델과 동일 방식)
        masked = attention_mask.unsqueeze(2) * embeddings  # [B, L, H]
        mask_len = torch.sum(attention_mask, dim=1, keepdim=True)  # [B, 1]
        pooled_rep = torch.sum(masked, dim=1) / (mask_len + 1e-8)      # [B, H]

        # D. 게이팅 네트워크로부터 8개 전문가에 대한 가중치 산출
        gate_weights = self.gating(pooled_rep)  # [B, 8]

        # E. 8개 서브모델의 출력 스택 및 가중 합산
        outputs = torch.stack([out1, out2, out3, out4, out5, out6, out7, out8], dim=1)  # [B, 8, H]
        ensemble_output = torch.sum(outputs * gate_weights.unsqueeze(-1), dim=1)  # [B, H]

        # F. 최종 분류
        logits = self.fc(ensemble_output)
        return logits
