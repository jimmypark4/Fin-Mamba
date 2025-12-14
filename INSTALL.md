# Fin-Mamba 설치 가이드 (Windows WSL2)

Windows WSL2 환경에서 Fin-Mamba 프로젝트를 처음부터 설치하고 실행하는 완전한 가이드입니다.

---

## 목차
1. [사전 요구사항](#1-사전-요구사항)
2. [WSL2 설치](#2-wsl2-설치)
3. [NVIDIA 드라이버 및 CUDA 설정](#3-nvidia-드라이버-및-cuda-설정)
4. [Miniconda 설치](#4-miniconda-설치)
5. [프로젝트 클론](#5-프로젝트-클론)
6. [Python 환경 설정](#6-python-환경-설정)
7. [프로젝트 실행](#7-프로젝트-실행)
8. [문제 해결](#8-문제-해결)

---

## 1. 사전 요구사항

| 항목 | 최소 요구사항 |
|------|--------------|
| OS | Windows 10 (버전 2004 이상) 또는 Windows 11 |
| GPU | NVIDIA GPU (RTX 3090 이상 권장) |
| RAM | 16GB 이상 |
| 저장공간 | 50GB 이상 여유 공간 |

---

## 2. WSL2 설치

### 2.1 PowerShell 관리자 모드 실행
Windows 검색에서 "PowerShell" → 우클릭 → "관리자 권한으로 실행"

### 2.2 WSL 설치
```powershell
wsl --install
```

### 2.3 재부팅 후 Ubuntu 설정
재부팅하면 자동으로 Ubuntu 설치가 시작됩니다.
- 사용자 이름 입력 (예: `jimmy`)
- 비밀번호 설정

### 2.4 WSL2 버전 확인
```powershell
wsl --list --verbose
```
VERSION이 2인지 확인합니다.

---

## 3. NVIDIA 드라이버 및 CUDA 설정

### 3.1 Windows에 NVIDIA 드라이버 설치
1. [NVIDIA 드라이버 다운로드](https://www.nvidia.com/Download/index.aspx) 접속
2. GPU에 맞는 최신 Game Ready 또는 Studio 드라이버 설치
3. **중요**: WSL2에서는 Windows 드라이버만 설치하면 됩니다. WSL 내부에 별도 드라이버 설치 불필요!

### 3.2 WSL에서 GPU 확인
```bash
nvidia-smi
```
GPU 정보가 출력되면 성공입니다.

---

## 4. Miniconda 설치

### 4.1 WSL Ubuntu 터미널 실행
Windows Terminal 또는 Ubuntu 앱 실행

### 4.2 Miniconda 다운로드 및 설치
```bash
# 홈 디렉토리로 이동
cd ~

# Miniconda 설치 스크립트 다운로드
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 설치 실행
bash Miniconda3-latest-Linux-x86_64.sh
```

설치 중 질문:
- `Do you accept the license terms?` → `yes`
- `Miniconda3 will now be installed into this location` → Enter (기본 경로 사용)
- `Do you wish to update your shell profile to automatically initialize conda?` → `yes`

### 4.3 터미널 재시작
```bash
source ~/.bashrc
```

### 4.4 Conda 설치 확인
```bash
conda --version
```

---

## 5. 프로젝트 클론

### 5.1 Git 설치 (없는 경우)
```bash
sudo apt update
sudo apt install git -y
```

### 5.2 프로젝트 클론
```bash
cd ~
git clone https://github.com/jimmypark4/Fin-Mamba.git
cd Fin-Mamba
```

---

## 6. Python 환경 설정

### 방법 A: environment.yml 사용 (권장)

```bash
# Conda 환경 생성 (약 10-15분 소요)
conda env create -f environment.yml

# 환경 활성화
conda activate finmamba

# 설치 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### 방법 B: 수동 설치

```bash
# 1. Conda 환경 생성
conda create -n finmamba python=3.12 -y
conda activate finmamba

# 2. PyTorch 설치 (CUDA 12.4)
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y

# 3. 핵심 의존성 설치
pip install transformers==4.48.0 hydra-core==1.3.2 hydra-optuna-sweeper==1.2.0
pip install datasets==3.2.0 pandas==2.2.3 scikit-learn==1.6.1
pip install einops==0.8.0 accelerate==1.3.0 peft==0.14.0
pip install omegaconf==2.3.0 tqdm==4.67.1

# 4. Mamba SSM 설치 (순서 중요!)
pip install ninja==1.11.1.3
pip install causal-conv1d==1.5.0.post8
pip install mamba-ssm==2.2.4

# 5. 시각화 도구
pip install matplotlib==3.10.0 seaborn==0.13.2
```

### 방법 C: requirements.txt 사용

```bash
# Conda 환경 생성
conda create -n finmamba python=3.12 -y
conda activate finmamba

# PyTorch 먼저 설치 (Conda 권장)
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y

# 나머지 패키지 설치
pip install -r requirements.txt
```

---

## 7. 프로젝트 실행

### 7.1 기본 실행 (FinBERT-Mamba3 + PhraseBank)
```bash
cd ~/Fin-Mamba
conda activate finmamba
python main.py
```

### 7.2 다른 모델로 실행
```bash
# LoRA 모델
python main.py model=finbert_lora

# Gamba7 앙상블 모델
python main.py model=gamba7
```

### 7.3 다른 데이터셋으로 실행
```bash
# FiQA 데이터셋
python main.py data.csv_path="./data/data_fiqa1.csv"

# Tweet 데이터셋
python main.py data.csv_path="./data/data_tweet.csv"
```

### 7.4 하이퍼파라미터 조정
```bash
# 학습률 변경
python main.py training.learning_rate=1e-5

# 에폭 수 변경
python main.py training.epochs=20

# 배치 사이즈 변경
python main.py data.batch_size=16

# 어댑터 레이어 수 변경
python main.py model.config.mamba_block.num_adapter_layers=10
```

### 7.5 Optuna 하이퍼파라미터 탐색
```bash
python main.py optuna.enabled=True optuna.n_trials=50
```

### 7.6 결과 확인
```bash
# 최신 결과 디렉토리 확인
ls -lt outputs/

# 결과 CSV 확인
cat outputs/YYYY-MM-DD/HH-MM-SS/results.csv
```

---

## 8. 문제 해결

### Q1: `nvidia-smi` 명령이 작동하지 않음
```bash
# Windows에서 NVIDIA 드라이버 재설치
# WSL 재시작
wsl --shutdown
wsl
```

### Q2: CUDA out of memory 에러
```bash
# 배치 사이즈 줄이기
python main.py data.batch_size=4

# GPU 메모리 확인
nvidia-smi
```

### Q3: `mamba-ssm` 설치 실패
```bash
# ninja 먼저 설치
pip install ninja

# CUDA 환경변수 확인
echo $CUDA_HOME

# 재설치
pip install causal-conv1d==1.5.0.post8 --no-cache-dir
pip install mamba-ssm==2.2.4 --no-cache-dir
```

### Q4: `ModuleNotFoundError: No module named 'xxx'`
```bash
# Conda 환경 활성화 확인
conda activate finmamba

# 패키지 설치 확인
pip list | grep xxx
```

### Q5: WSL에서 Windows 파일 접근
```bash
# Windows C 드라이브 접근
cd /mnt/c/Users/YourName/

# Windows에서 WSL 파일 접근 (탐색기 주소창)
\\wsl$\Ubuntu\home\jimmy\Fin-Mamba
```

### Q6: Hydra 설정 오류
```bash
# 설정 파일 문법 확인
python main.py --cfg job

# 기본값으로 실행
python main.py
```

---

## 빠른 시작 (전체 명령어)

WSL Ubuntu가 설치되어 있고 NVIDIA 드라이버가 작동하는 상태에서:

```bash
# 1. Miniconda 설치
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
source ~/.bashrc

# 2. 프로젝트 클론
git clone https://github.com/jimmypark4/Fin-Mamba.git
cd Fin-Mamba

# 3. 환경 설정
conda env create -f environment.yml
conda activate finmamba

# 4. 실행
python main.py
```

---

## 유용한 명령어 모음

```bash
# Conda 환경 목록
conda env list

# 환경 활성화/비활성화
conda activate finmamba
conda deactivate

# GPU 모니터링 (실시간)
watch -n 1 nvidia-smi

# 학습 로그 실시간 확인
tail -f outputs/YYYY-MM-DD/HH-MM-SS/main.log

# 환경 삭제 (필요시)
conda env remove -n finmamba
```
