# src/training/trainer_factory.py
from .bert_trainer import BertTrainer
from .mamba_trainer import MambaTrainer
from .finbert_trainer import FinBertTrainer

def get_trainer(model, cfg):
    """
    모델 종류/설정(cfg)에 따라 적절한 Trainer 클래스를 인스턴스화해 반환
    """
    if cfg.model.config.embedding == "state-spaces/mamba-130m-hf":
        
        return MambaTrainer(model, cfg)
    elif cfg.model.config.embedding == "bert-base-uncased":
        return BertTrainer(model, cfg)
    elif cfg.model.config.embedding == "ProsusAI/finbert":
        return FinBertTrainer(model, cfg)
    else:
        raise ValueError("지원하지 않는 훈련 임베딩 모델입니다.")
