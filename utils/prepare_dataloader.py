from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader
from utils.mamba_dataset import MambaDataset  # MambaDataset 클래스
from utils.text_dataset import TextDataset  # TextDataset 클래스
from utils.finbert_dataset import FinBertTextDataset  # FinBertTextDataset 클래스
def prepare_dataloaders(cfg: DictConfig):
    """
    1) CSV 로드
    2) train/test 분할
    3) DataLoader 반환 (train, valid, test)
    """
    # CSV 데이터 로드
    df = pd.read_csv(cfg.data.csv_path)

    # 데이터셋 형식 확인 및 열 이름 검증
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV 파일에 'text' 또는 'label' 열이 포함되어 있지 않습니다.")

    # train/test 분할
    train_df, temp_df = train_test_split(
        df, test_size=cfg.data.test_size + cfg.data.valid_size, random_state=cfg.experiment.seed
    )
    valid_df, test_df = train_test_split(
        temp_df, test_size=cfg.data.test_size / (cfg.data.test_size + cfg.data.valid_size), random_state=cfg.experiment.seed
    )

    # 동적으로 데이터셋 클래스 선택
    if cfg.model.config.embedding == "state-spaces/mamba-130m-hf":  # 수정된 부분
        DatasetClass = MambaDataset
    elif cfg.model.config.embedding == "bert-base-uncased":
        DatasetClass = TextDataset    
    elif cfg.model.config.embedding == "ProsusAI/finbert":
        DatasetClass = FinBertTextDataset
    else:
        raise ValueError("지원하지 않는 임베딩 모델입니다.")
    # 데이터셋 생성
    train_dataset = DatasetClass(train_df)
    valid_dataset = DatasetClass(valid_df)

    test_dataset = DatasetClass(test_df)
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers
    )

    # valid_loader를 None으로 설정
    return train_loader, valid_loader, test_loader
