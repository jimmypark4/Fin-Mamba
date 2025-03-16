import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba, Mamba2
from transformers import AutoModelForCausalLM, AutoConfig
# xgboost 임포트는 이번 예시에서는 사용하지 않습니다.

#############################################
# 공통 레이어 및 블록 정의
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
    MambaBlock: RMSNorm -> Mamba2 -> RMSNorm -> MLP
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

# BaseSubModel: 서브모델들의 공통 인터페이스
class BaseSubModel(nn.Module):
    def forward(self, embeddings, attention_mask):
        raise NotImplementedError("서브클래스에서 구현하세요.")

#############################################
# 5개의 서브모델 (모두 MambaBlock만 사용)
#############################################

class SubModel1(BaseSubModel):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList()
        d_model   = config.mamba_block.d_model
        d_state   = config.mamba_block.d_state
        d_conv    = config.mamba_block.d_conv
        expand    = config.mamba_block.expand
        eps       = config.rmsnorm_eps
        num_cycles= config.num_cycles

        for _ in range(num_cycles):
            for _ in range(6):
                self.blocks.append(
                    MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, rmsnorm_eps=eps)
                )

    def forward(self, embeddings, attention_mask):
        # 마스크 평균 풀링
        masked = attention_mask.unsqueeze(2) * embeddings  # [B, L, H]
        mask_len = torch.sum(attention_mask, dim=1, keepdim=True)  # [B, 1]
        pooled = torch.sum(masked, dim=1) / (mask_len + 1e-8)  # [B, H]
        # [B, 1, H]로 변환 후 블록 적용
        x = pooled.unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        return x.squeeze(1)  # [B, H]

class SubModel2(BaseSubModel):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList()
        d_model   = config.mamba_block.d_model
        d_state   = config.mamba_block.d_state
        d_conv    = config.mamba_block.d_conv
        expand    = config.mamba_block.expand
        eps       = config.rmsnorm_eps
        num_cycles= config.num_cycles

        for _ in range(num_cycles):
            for _ in range(6):
                self.blocks.append(
                    MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, rmsnorm_eps=eps)
                )

    def forward(self, embeddings, attention_mask):
        masked = attention_mask.unsqueeze(2) * embeddings
        mask_len = torch.sum(attention_mask, dim=1, keepdim=True)
        pooled = torch.sum(masked, dim=1) / (mask_len + 1e-8)
        x = pooled.unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        return x.squeeze(1)

class SubModel3(BaseSubModel):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList()
        d_model   = config.mamba_block.d_model
        d_state   = config.mamba_block.d_state
        d_conv    = config.mamba_block.d_conv
        expand    = config.mamba_block.expand
        eps       = config.rmsnorm_eps
        num_cycles= config.num_cycles

        for _ in range(num_cycles):
            for _ in range(6):
                self.blocks.append(
                    MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, rmsnorm_eps=eps)
                )

    def forward(self, embeddings, attention_mask):
        masked = attention_mask.unsqueeze(2) * embeddings
        mask_len = torch.sum(attention_mask, dim=1, keepdim=True)
        pooled = torch.sum(masked, dim=1) / (mask_len + 1e-8)
        x = pooled.unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        return x.squeeze(1)

class SubModel4(BaseSubModel):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList()
        d_model   = config.mamba_block.d_model
        d_state   = config.mamba_block.d_state
        d_conv    = config.mamba_block.d_conv
        expand    = config.mamba_block.expand
        eps       = config.rmsnorm_eps
        num_cycles= config.num_cycles

        for _ in range(num_cycles):
            for _ in range(6):
                self.blocks.append(
                    MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, rmsnorm_eps=eps)
                )

    def forward(self, embeddings, attention_mask):
        masked = attention_mask.unsqueeze(2) * embeddings
        mask_len = torch.sum(attention_mask, dim=1, keepdim=True)
        pooled = torch.sum(masked, dim=1) / (mask_len + 1e-8)
        x = pooled.unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        return x.squeeze(1)

class SubModel5(BaseSubModel):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList()
        d_model   = config.mamba_block.d_model
        d_state   = config.mamba_block.d_state
        d_conv    = config.mamba_block.d_conv
        expand    = config.mamba_block.expand
        eps       = config.rmsnorm_eps
        num_cycles= config.num_cycles

        for _ in range(num_cycles):
            for _ in range(6):
                self.blocks.append(
                    MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, rmsnorm_eps=eps)
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
        self.temp = init_temp  # 초기 temperature 값

    def forward(self, pooled_rep):
        # pooled_rep: [B, d_model]
        logits = self.linear(pooled_rep)  # [B, num_experts]
        # Gumbel-Softmax 적용: hard=True이면 one-hot 형태에 가깝게 출력됨
        weights = F.gumbel_softmax(logits, tau=self.temp, hard=True, dim=-1)  # [B, num_experts]
        return weights

#############################################
# Gamba5: Mamba LM -> 5 서브모델 병렬 -> 게이팅 네트워크를 통한 앙상블 -> 최종 분류
#############################################

class Gamba5(nn.Module):
    """
    구조:
    Mamba LM -> (SubModel1, SubModel2, SubModel3, SubModel4, SubModel5) 병렬 처리 ->
    게이팅 네트워크를 통한 가중 합산 -> 최종 분류
    """
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        
        # 1) 사전학습된 Mamba LM 설정
        embedding = "state-spaces/mamba-130m-hf"
        self.mamba_config = AutoConfig.from_pretrained(embedding)
        self.mamba_config.output_hidden_states = True
        self.mamba_model = AutoModelForCausalLM.from_pretrained(embedding, config=self.mamba_config)
        mamba_hidden_size = self.mamba_config.hidden_size
        
        # Mamba LM의 hidden size를 서브모델과 맞춤
        config.mamba_block.d_model = mamba_hidden_size
        
        # 2) 5개의 서브모델 생성 (모두 MambaBlock 기반)
        self.sub_model1 = SubModel1(config)
        self.sub_model2 = SubModel2(config)
        self.sub_model3 = SubModel3(config)
        self.sub_model4 = SubModel4(config)
        self.sub_model5 = SubModel5(config)
        
        # 3) 게이팅 네트워크 (pooled representation을 입력으로 받아 5개 전문가에 대한 가중치 산출)
        self.gating = GatingNetwork(d_model=mamba_hidden_size, num_experts=5, init_temp=1.0)
        
        # 4) 최종 분류 레이어 (입력 차원: mamba_hidden_size)
        self.fc = nn.Linear(mamba_hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask):
        """
        input_ids: [B, L]
        attention_mask: [B, L]
        """
        # A. Mamba LM을 통해 임베딩 추출 (hidden_states[-1]: [B, L, hidden_dim])
        mamba_output = self.mamba_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        embeddings = mamba_output.hidden_states[-1]  # [B, L, H]
        
        # B. 각 서브모델에 임베딩과 attention_mask 전달하여 특성 추출
        out1 = self.sub_model1(embeddings, attention_mask)  # [B, H]
        out2 = self.sub_model2(embeddings, attention_mask)
        out3 = self.sub_model3(embeddings, attention_mask)
        out4 = self.sub_model4(embeddings, attention_mask)
        out5 = self.sub_model5(embeddings, attention_mask)
        
        # C. 게이팅 네트워크를 위한 pooled representation (서브모델과 동일한 방식 사용)
        masked = attention_mask.unsqueeze(2) * embeddings  # [B, L, H]
        mask_len = torch.sum(attention_mask, dim=1, keepdim=True)  # [B, 1]
        pooled_rep = torch.sum(masked, dim=1) / (mask_len + 1e-8)  # [B, H]
        
        # D. 게이팅 네트워크로부터 가중치 산출 (Gumbel-Softmax 적용)
        gate_weights = self.gating(pooled_rep)  # [B, 5]
        
        # E. 5개 서브모델의 출력을 스택 -> [B, 5, H]
        outputs = torch.stack([out1, out2, out3, out4, out5], dim=1)
        # F. 게이팅 가중치를 이용해 weighted sum (각 샘플별로 한 전문가에 거의 할당됨)
        ensemble_output = torch.sum(outputs * gate_weights.unsqueeze(-1), dim=1)  # [B, H]
        
        # G. 최종 분류
        logits = self.fc(ensemble_output)
        return logits
