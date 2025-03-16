import hydra
from omegaconf import DictConfig, OmegaConf
import os
import logging
import optuna

import torch
from src.training.trainer import Trainer
from src.testing.evaluator import Evaluator
from utils.prepare_dataloader import prepare_dataloaders
from src.training.trainer_factory import get_trainer
from src.testing.evaluator_factory import get_evaluator


"""
 dropout_rate: 0.1
  rmsnorm_eps: 1e-6
  mamba_block:
    d_state: 16
    d_conv: 4
    expand: 2
    num_adapter_layers: 2  # Adapter로 쌓을 MambaBlock 계층 수
  adapter_gate_dim: 768  # 보통 d_model과 동일하게 설정

"""
def objective(trial: optuna.Trial, cfg: DictConfig) -> float:
    logging.info("Objective config:\n" + OmegaConf.to_yaml(cfg))

    OmegaConf.set_struct(cfg.experiment, False)
    
    cfg.experiment.seed = trial.suggest_int("seed", 0, 100)

    OmegaConf.set_struct(cfg.model.config, False)
    # 하이퍼파라미터 샘플링
    # cfg.model.config.dropout_rate = trial.suggest_categorical("dropout_rate", [0.1, 0.2, 0.3])
    # cfg.model.config.rmsnorm_eps = trial.suggest_categorical("rmsnorm_eps", [1e-6, 1e-5, 1e-4, 1e-3])
    cfg.model.config.mamba_block.num_adapter_layers = trial.suggest_int("num_adapter_layers", 1, 25)

    OmegaConf.set_struct(cfg.training, False)
    cfg.training.learning_rate = trial.suggest_categorical("learning_rate", [1e-6, 1e-5, 1e-4, 1e-3])
    # cfg.training.epoch = trial.suggest_categorical("epoch", [5, 10,20,30])

    # 데이터 로딩 (test_loader는 평가용)
    train_loader, valid_loader, test_loader = prepare_dataloaders(cfg)
    
    # 모델 인스턴스화 및 학습
    model = hydra.utils.instantiate(cfg.model)
    trainer = get_trainer(model, cfg)
    trainer.train(train_loader, valid_loader)  
    # 평가 (test_loader 사용)
    evaluator = get_evaluator(model, cfg)
    metrics = evaluator.evaluate(test_loader)
    
    return metrics["accuracy"]

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logging.info("start main execution")
    logging.info("Full Config:\n" + OmegaConf.to_yaml(cfg))

    # logging.info(OmegaConf.to_yaml(cfg.model))
    os.makedirs(cfg.experiment.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.experiment.metric_dir, exist_ok=True)
    
    # Optuna를 사용할지 여부에 따라 분기
    if cfg.optuna.enabled:
        study = optuna.create_study(direction="maximize")  # 정확도를 최대화하는 방향으로 최적화
        study.optimize(lambda trial: objective(trial, cfg), n_trials=cfg.optuna.n_trials)
        logging.info("Best trial:")
        best_trial = study.best_trial
        logging.info(f"Best acc: {best_trial.value * 100:.2f}%")
        logging.info("Best hyperparameters:")
    
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        best_config = OmegaConf.create(cfg_dict)
        logging.info("Full Config with Best Hyperparameters:\n" + OmegaConf.to_yaml(best_config))
    else:
        # 데이터 준비
        train_loader, valid_loader, test_loader = prepare_dataloaders(cfg)
        # 모델 인스턴스화
        model = hydra.utils.instantiate(cfg.model)
        # 학습
        trainer = get_trainer(model, cfg)
        trainer.train(train_loader, valid_loader)  # 학습은 train_loader, 평가용 test_loader 사용
        # 테스트 및 결과 저장
        evaluator = get_evaluator(model, cfg)
        metrics = evaluator.evaluate(test_loader)
        with open("metrics.json", "w") as f:
            import json
            json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
