# Fin-Mamba: Parameter-Efficient Mamba Adapters for Financial Sentiment Analysis

금융 텍스트 감성 분석을 위한 Mamba 기반 Parameter-Efficient Fine-Tuning (PEFT) 연구 프로젝트입니다.

## 연구 개요

### 연구 목적
본 연구는 State Space Model (SSM) 기반의 **Mamba 아키텍처**를 금융 NLP 태스크에 적용하여, 기존 Transformer 기반 모델 대비 효율적인 파인튜닝 방법을 탐구합니다.

### 핵심 기여
1. **FinBERT-Mamba Adapter**: 사전학습된 FinBERT에 Mamba 블록 기반 어댑터를 추가하여 파라미터 효율적 학습
2. **Gated Adapter Fusion**: 다중 어댑터 레이어의 출력을 동적으로 융합하는 게이팅 메커니즘 제안
3. **Gamba (Gated Mamba Ensemble)**: Mamba와 Attention 블록을 결합한 Expert 앙상블 아키텍처
4. **Ablation Study**: 어댑터 레이어 수(1-60)에 따른 성능 변화 분석

## 프로젝트 구조

```
Fin-Mamba/
├── conf/                          # Hydra 설정 파일
│   ├── config.yaml               # 메인 설정
│   └── model/                    # 모델별 설정 (26개)
│       ├── finbert_mamba3.yaml   # 기본 모델 설정
│       ├── gamba7.yaml           # Gamba 앙상블 설정
│       └── ...
├── data/                          # 데이터셋
│   ├── data_phrasebank.csv       # Financial PhraseBank (4,847 samples)
│   ├── data_fiqa1.csv            # FiQA Task 1 (823 samples)
│   ├── data_mltreve23.csv        # ML-TRev 2023 (5,843 samples)
│   └── data_tweet.csv            # Financial Tweets (9,937 samples)
├── src/
│   ├── models/                   # 모델 아키텍처 (15개)
│   │   ├── finbert_mamba3.py     # 핵심 모델: FinBERT + Gated Mamba Adapter
│   │   ├── gamba7.py             # Expert 앙상블 모델
│   │   ├── finbert_lora.py       # LoRA 베이스라인
│   │   └── ...
│   ├── training/                 # 학습 모듈
│   │   ├── trainer_factory.py    # 트레이너 팩토리
│   │   └── *_trainer.py          # 모델별 트레이너
│   └── testing/                  # 평가 모듈
│       ├── evaluator_factory.py  # 평가기 팩토리
│       └── *_evaluator.py        # 모델별 평가기
├── utils/                         # 유틸리티
│   ├── prepare_dataloader.py     # 데이터 로더 준비
│   ├── confusion_matrix.py       # 혼동 행렬 시각화
│   └── tsne.py                   # t-SNE 시각화
├── mamba/                         # Mamba SSM 소스 (참고용)
├── results/                       # 실험 결과 CSV
├── outputs/                       # Hydra 출력 (체크포인트, 로그)
├── main.py                        # 메인 실행 파일
├── run_abligation.sh             # Ablation 실험 스크립트
├── environment.yml               # Conda 환경 설정
└── requirements.txt              # pip 의존성 (참고용)
```

## 모델 아키텍처

### 1. FinBERT-Mamba3 (핵심 모델)

```
┌─────────────────────────────────────────────────────────────┐
│                      FinBERT (Frozen)                       │
│                           ↓                                 │
│                    [CLS] Embedding                          │
│                           ↓                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Stacked MambaBlock Adapters              │  │
│  │  ┌─────────┐   ┌─────────┐       ┌─────────┐        │  │
│  │  │ Mamba   │ → │ Mamba   │ → ... │ Mamba   │        │  │
│  │  │ Block 1 │   │ Block 2 │       │ Block N │        │  │
│  │  └────┬────┘   └────┬────┘       └────┬────┘        │  │
│  │       ↓             ↓                 ↓              │  │
│  │  ┌────────────────────────────────────────────────┐ │  │
│  │  │           Gating Network (Softmax)             │ │  │
│  │  │        Dynamic Adapter Fusion Weights          │ │  │
│  │  └────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ↓                                 │
│              Residual: [CLS] + Fused Adapter               │
│                           ↓                                 │
│                    Classification Head                      │
└─────────────────────────────────────────────────────────────┘
```

**MambaBlock 구조:**
```
RMSNorm → Mamba2 → Dropout → RMSNorm → MLP (GELU) → Dropout
          (Residual Connection)
```

### 2. Gamba7 (Expert Ensemble)

```
┌─────────────────────────────────────────────────────────────┐
│                  Mamba-130M (Embeddings)                    │
│                           ↓                                 │
│         ┌─────────────────┼─────────────────┐              │
│         ↓                 ↓                 ↓              │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐          │
│   │ SubModel1│     │ SubModel2│     │ SubModel3│          │
│   │ 3×Mamba  │     │3×Attention│    │  Mixed   │          │
│   └────┬─────┘     └────┬─────┘     └────┬─────┘          │
│        └────────────────┼────────────────┘                │
│                         ↓                                  │
│              Softmax Gating Network                        │
│                         ↓                                  │
│              Weighted Expert Ensemble                      │
│                         ↓                                  │
│                  Classification Head                       │
└─────────────────────────────────────────────────────────────┘
```

### 3. 지원 모델 목록

| 카테고리 | 모델명 | 설명 |
|---------|-------|------|
| **FinBERT 기반** | `finbert` | 베이스라인 (Full Fine-tuning) |
| | `finbert_mamba` | FinBERT + Single Mamba Adapter |
| | `finbert_mamba2` | Enhanced Mamba Adapter |
| | `finbert_mamba3` | **Gated Multi-layer Adapter (핵심)** |
| | `finbert_mamba_nogate` | Gating 없는 버전 |
| | `finbert_attention` | Attention 기반 Adapter |
| | `finbert_lora` | LoRA Adapter |
| | `finbert_qlora` | Quantized LoRA |
| **Mamba 기반** | `mamba3b` | Mamba LM 직접 분류 |
| | `mamba_lstm` | Mamba + Bi-LSTM |
| | `mamba_blocks` | Stacked Mamba Blocks |
| **Gamba** | `gamba` ~ `gamba7` | Expert Ensemble 변형들 |
| **베이스라인** | `bert_lstm` | BERT + LSTM |
| | `cnn_lstm` | CNN + LSTM |

## 환경 설정

### 요구사항
- Python 3.12
- CUDA 12.4+ (Mamba SSM 연산에 필요)
- GPU: NVIDIA GPU (권장: RTX 3090 이상)

### 설치

```bash
# 1. Conda 환경 생성 (environment.yml 사용)
conda env create -f environment.yml
conda activate finmamba

# 2. 실행 확인
python main.py --help
```

#### 수동 설치 (environment.yml 없이)

```bash
# 1. Conda 환경 생성
conda create -n finmamba python=3.12
conda activate finmamba

# 2. PyTorch 설치 (CUDA 12.4)
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia

# 3. 핵심 의존성 설치
pip install transformers==4.48.0 hydra-core==1.3.2 hydra-optuna-sweeper==1.2.0
pip install datasets==3.2.0 pandas==2.2.3 scikit-learn==1.6.1
pip install einops==0.8.0 accelerate==1.3.0 peft==0.14.0

# 4. Mamba SSM 설치
pip install causal-conv1d==1.5.0.post8
pip install mamba-ssm==2.2.4

# 5. 시각화 도구 (선택)
pip install matplotlib==3.10.0 seaborn==0.13.2
```

### 주요 의존성
- `torch==2.5.1` (CUDA 12.4)
- `transformers==4.48.0`
- `mamba-ssm==2.2.4`
- `causal-conv1d==1.5.0.post8`
- `hydra-core==1.3.2`
- `optuna==2.10.1`
- `triton==3.1.0`

## 실행 방법

### 기본 학습

```bash
# 기본 설정으로 학습 (FinBERT-Mamba3, PhraseBank)
python main.py

# 특정 모델로 학습
python main.py model=gamba7

# 특정 데이터셋으로 학습
python main.py data.csv_path="./data/data_fiqa1.csv"

# 하이퍼파라미터 변경
python main.py training.learning_rate=1e-5 training.epochs=20
```

### 어댑터 레이어 수 변경

```bash
# 어댑터 레이어 수 조절
python main.py model.config.mamba_block.num_adapter_layers=10
```

### Optuna 하이퍼파라미터 탐색

```bash
python main.py optuna.enabled=True optuna.n_trials=50
```

### Ablation Study 실행

```bash
# 어댑터 레이어 수(1-60)에 대한 ablation
bash run_abligation2.sh
```

## 설정 파일 구조

### config.yaml (메인 설정)

```yaml
defaults:
  - model: finbert_mamba3      # 모델 선택

experiment:
  seed: 32
  checkpoint_dir: "${hydra:run.dir}/checkpoints"
  result_csv: "${hydra:run.dir}/results.csv"

data:
  csv_path: "./data/data_phrasebank.csv"
  batch_size: 8
  test_size: 0.2
  valid_size: 0.1

training:
  epochs: 10
  learning_rate: 1e-6
  early_stopping_patience: 2
  weight_decay: 0.01

optuna:
  enabled: False
  n_trials: 1
```

### 모델 설정 예시 (finbert_mamba3.yaml)

```yaml
_target_: src.models.finbert_mamba3.FinBERT_Mamba3

config:
  embedding: "ProsusAI/finbert"
  num_labels: 3
  dropout_rate: 0.2
  rmsnorm_eps: 1e-6
  mamba_block:
    d_state: 16
    d_conv: 4
    expand: 2
    num_adapter_layers: 23    # 어댑터 레이어 수
  adapter_gate_dim: 768
```

## 실험 결과

### PhraseBank 데이터셋 성능 (Adapter Layer 수 변화)

| Adapter Layers | Accuracy | F1-Score |
|:--------------:|:--------:|:--------:|
| 1 | 90.10% | 89.40% |
| 2 | 90.62% | 89.97% |
| 6 | 90.72% | 89.79% |
| 11 | 90.82% | 89.81% |
| **18** | **91.24%** | **90.61%** |
| 21 | 91.03% | 90.32% |
| 32 | 90.93% | 90.07% |
| 42 | 91.03% | 89.86% |

**주요 발견:**
- 어댑터 레이어 수 18개에서 최고 성능 달성
- 레이어 수 증가가 항상 성능 향상으로 이어지지 않음
- 최적 레이어 수는 데이터셋에 따라 다를 수 있음

### 데이터셋 정보

| 데이터셋 | 샘플 수 | 출처 | 설명 |
|---------|--------|------|------|
| PhraseBank | 4,847 | Financial PhraseBank | 금융 문장 감성 |
| FiQA-1 | 823 | FiQA Challenge | 금융 QA 감성 |
| ML-TRev23 | 5,843 | ML-TRev 2023 | 금융 리뷰 |
| Tweet | 9,937 | Financial Tweets | 금융 트윗 감성 |

**레이블:** 0 (Negative), 1 (Neutral), 2 (Positive)

## 출력 결과물

### 학습 결과 저장 위치
```
outputs/
└── YYYY-MM-DD/
    └── HH-MM-SS/
        ├── .hydra/
        │   ├── config.yaml      # 실행 시 전체 설정
        │   └── overrides.yaml   # CLI 오버라이드
        ├── checkpoints/         # 에폭별 모델 체크포인트
        └── results.csv          # 평가 결과
```

### 결과 CSV 형식
```csv
dataset,num_layer,accuracy,f1
data_phrasebank,18,0.9124,0.9061
```

## 코드 설명

### 핵심 컴포넌트

#### 1. MambaBlock (src/models/finbert_mamba3.py)
```python
class MambaBlock(nn.Module):
    """RMSNorm → Mamba2 → Dropout → RMSNorm → MLP"""
    def __init__(self, d_model, d_state, d_conv, expand, dropout_rate, rmsnorm_eps):
        self.norm1 = RMSNorm(d_model)
        self.mamba = Mamba2(d_model, d_state, d_conv, expand)
        self.mlp = nn.Sequential(Linear, GELU, Dropout, Linear, Dropout)
```

#### 2. Gated Adapter Fusion
```python
# 각 어댑터 레이어 출력을 스택
stacked_adapters = torch.stack(adapter_outputs, dim=1)  # [B, num_layers, d_model]

# 게이팅 네트워크로 가중치 계산
gate_weights = self.gate_activation(self.adapter_gate(cls_output))  # [B, num_layers]

# 동적 융합
fused_adapter = torch.sum(stacked_adapters * gate_weights.unsqueeze(-1), dim=1)
```

#### 3. Trainer Factory (src/training/trainer_factory.py)
```python
def get_trainer(model, cfg):
    embedding = cfg.model.config.embedding
    if "finbert" in embedding:
        return FinBertTrainer(model, cfg)
    elif "mamba" in embedding:
        return MambaTrainer(model, cfg)
```

### 데이터 파이프라인
1. CSV 로드 (`text`, `label` 컬럼)
2. Train/Valid/Test 분할 (70%/10%/20%)
3. 모델에 맞는 토크나이저 적용 (FinBERT: max_length=32)
4. DataLoader 생성 (batch_size=8)

## 인수인계 시 주의사항

### 1. 환경 설정
- Mamba SSM은 GPU가 필수 (CUDA 12.4 이상)
- `conda env create -f environment.yml`로 환경 생성
- CUDA 버전과 PyTorch 버전 호환성 확인 필요

### 2. 실험 재현
- `experiment.seed` 값을 고정하여 재현성 확보
- Hydra가 자동으로 설정을 `outputs/`에 저장하므로 과거 실험 재현 가능

### 3. 모델 확장
- 새 모델 추가 시:
  1. `src/models/`에 모델 클래스 구현
  2. `conf/model/`에 YAML 설정 추가
  3. 필요시 `trainer_factory.py`, `evaluator_factory.py` 수정

### 4. 데이터셋 추가
- `data/` 디렉토리에 CSV 추가 (컬럼: `text`, `label`)
- `config.yaml`에서 `data.csv_path` 수정

## 참고 문헌

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [FinBERT: Financial Sentiment Analysis with Pre-trained Language Models](https://arxiv.org/abs/1908.10063)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

## 라이선스

연구 목적으로 개발되었습니다.

## 문의

프로젝트 관련 문의사항이 있으시면 담당자에게 연락해 주세요.
