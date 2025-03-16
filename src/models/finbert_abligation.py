import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

# 간단한 Adapter: configurable MLP 기반 Bottleneck 구조
class SimpleAdapter(nn.Module):
    def __init__(self, d_model, adapter_dim=64, dropout_rate=0.1, num_mlp_layers=2):
        """
        Args:
            d_model (int): 원래 모델의 hidden dimension.
            adapter_dim (int): bottleneck 차원.
            dropout_rate (float): dropout 비율.
            num_mlp_layers (int): MLP 내에 사용될 레이어의 총 개수. 최소 2이어야 함.
                                    2인 경우, 단순히 down, up 프로젝션 (기존 구조와 동일).
                                    3 이상이면 중간에 adapter_dim → adapter_dim 레이어가 추가됨.
        """
        super().__init__()
        assert num_mlp_layers >= 2, "num_mlp_layers는 최소 2 이상이어야 합니다."
        layers = []
        # 첫 번째 레이어: d_model -> adapter_dim
        layers.append(nn.Linear(d_model, adapter_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        # 중간 레이어들: adapter_dim -> adapter_dim (num_mlp_layers - 2개)
        for _ in range(num_mlp_layers - 2):
            layers.append(nn.Linear(adapter_dim, adapter_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        # 마지막 레이어: adapter_dim -> d_model
        layers.append(nn.Linear(adapter_dim, d_model))
        layers.append(nn.Dropout(dropout_rate))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.mlp(x)

# FinBERT 모델에 간단한 Adapter를 적용한 PEFT 모델
class FinBERT_SimpleAdapter(nn.Module):
    def __init__(self, config):
        """
        config:
          - config.embedding: FinBERT 모델 경로 (예: "ProsusAI/finbert")
          - config.num_labels: 분류 클래스 수
          - config.dropout_rate: Adapter의 dropout 비율 (예: 0.1)
          - config.adapter_dim: Adapter 내부 bottleneck 차원 (예: 64 또는 32)
          - config.num_mlp_layers: MLP 내 레이어 수 (예: 2, 3, 4 등)
        """
        super().__init__()
        self.num_labels = config.num_labels
        self.embedding = config.embedding

        # 사전학습된 FinBERT 모델 로드 (hidden_states 포함)
        self.config_model = AutoConfig.from_pretrained(self.embedding, output_hidden_states=True)
        self.finetuned_model = AutoModel.from_pretrained(self.embedding, config=self.config_model)

        # PEFT 방식: 사전학습 모델 파라미터 동결
        for param in self.finetuned_model.parameters():
            param.requires_grad = False

        d_model = self.config_model.hidden_size

        # SimpleAdapter에 num_mlp_layers 값을 전달 (기본값은 2)
        self.adapter = SimpleAdapter(
            d_model=d_model,
            adapter_dim=config.adapter_dim,
            dropout_rate=config.dropout_rate,
            num_mlp_layers=getattr(config, "num_mlp_layers", 2)
        )

        # 최종 분류 헤드
        self.classifier = nn.Linear(d_model, self.num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.finetuned_model(input_ids=input_ids, attention_mask=attention_mask)
        # 마지막 hidden_states 사용, [CLS] 토큰 (첫 번째 토큰) 사용
        hidden_states = outputs.hidden_states[-1]  # [B, L, d_model]
        cls_output = hidden_states[:, 0, :]          # [B, d_model]

        # Adapter를 통해 미세조정 가능한 부분만 업데이트
        adapter_output = self.adapter(cls_output)    # [B, d_model]
        # Residual 연결: 원래 [CLS] 임베딩 + adapter 출력
        combined = cls_output + adapter_output

        logits = self.classifier(combined)
        return logits
