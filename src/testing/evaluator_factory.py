# src/testing/evaluator_factory.py
from .bert_evaluator import BertEvaluator
from .mamba_evaluator import MambaEvaluator
from .finbert_evaluator import FinBertEvaluator

def get_evaluator(model, cfg):
    """
    config 파라미터에 따라 BertEvaluator / MambaEvaluator 인스턴스화
    """
    if cfg.model.config.embedding == "state-spaces/mamba-130m-hf":
        return MambaEvaluator(model, cfg)
    elif cfg.model.config.embedding == "bert-base-uncased":
        return BertEvaluator(model, cfg)
    elif cfg.model.config.embedding == "ProsusAI/finbert":
        return FinBertEvaluator(model, cfg)
    else :
        raise ValueError("지원하지 않는 임베딩 모델입니다.")