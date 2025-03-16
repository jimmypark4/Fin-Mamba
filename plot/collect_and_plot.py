import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt

def parse_main_log(log_path):
    """
    main.log에서
    [Evaluator] Test Loss: X, Test Acc: Y, F1: Z
    형태의 로그를 찾아서
    (test_loss, test_acc, f1) 튜플로 반환
    """
    test_loss = None
    test_acc = None
    f1_score_val = None

    # 정규표현식 예시
    # "[Evaluator] Test Loss: ([\d\.]+).*Test Acc: ([\d\.]+).*%.*F1:\s+([\d\.]+)"
    # 주의: Acc=83.71% 처럼 '%' 포함
    pattern = re.compile(
        r"\[Evaluator\].*Test Loss:\s*([\d\.]+).*Test Acc:\s*([\d\.]+).*%.*F1:\s*([\d\.]+)",
        re.IGNORECASE
    )

    if not os.path.exists(log_path):
        return None, None, None

    with open(log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                # 예: X=0.4229, Y=83.71, Z=0.8119
                test_loss = float(match.group(1))
                test_acc = float(match.group(2))  # 83.71
                f1_score_val = float(match.group(3))
                break

    return test_loss, test_acc, f1_score_val

def parse_overrides(overrides_path):
    """
    .hydra/overrides.yaml에서 예:
    model.config.num_mamba_blocks=8
    형태의 설정을 찾고 정수 변환
    """
    num_blocks = None

    if not os.path.exists(overrides_path):
        return None

    with open(overrides_path, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            # 예: "model.config.num_mamba_blocks=16"
            if "model.config.num_mamba_blocks=" in line:
                # "model.config.num_mamba_blocks=16"
                parts = line.split("=")
                if len(parts) == 2:
                    val = parts[1].strip()
                    if val.isdigit():
                        num_blocks = int(val)
                    else:
                        # range(...) 등일 경우 추가 처리 필요할 수도 있음
                        pass
                break

    return num_blocks

def collect_results(root_dir):
    """
    root_dir 아래에 0,1,2,... 등의 실험 디렉터리가 있다고 가정.
    각 디렉터리:
      - main.log
      - .hydra/overrides.yaml
    """
    rows = []

    for run_name in os.listdir(root_dir):
        run_path = os.path.join(root_dir, run_name)
        if not os.path.isdir(run_path):
            continue

        # main.log 파일
        log_path = os.path.join(run_path, "main.log")
        # .hydra/overrides.yaml
        overrides_path = os.path.join(run_path, ".hydra", "overrides.yaml")

        if os.path.exists(log_path) and os.path.exists(overrides_path):
            # 파라미터(블록 수) 파싱
            num_blocks = parse_overrides(overrides_path)
            # 로그에서 메트릭 파싱
            test_loss, test_acc, f1_val = parse_main_log(log_path)

            if num_blocks is not None and test_loss is not None:
                rows.append({
                    "num_mamba_blocks": num_blocks,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "f1": f1_val
                })

    return pd.DataFrame(rows)

def plot_metrics(df):
    """
    df : columns=[num_mamba_blocks, test_loss, test_acc, f1]
    """
    df = df.sort_values("num_mamba_blocks")

    plt.figure(figsize=(12,4))

    # 1) Test Loss
    plt.subplot(1,3,1)
    plt.plot(df["num_mamba_blocks"], df["test_loss"], marker='o', color='blue')
    plt.xlabel("num_mamba_blocks")
    plt.ylabel("Test Loss")
    plt.title("Test Loss vs. #Blocks")
    plt.grid(True)

    # 2) Test Accuracy
    plt.subplot(1,3,2)
    plt.plot(df["num_mamba_blocks"], df["test_acc"], marker='o', color='red')
    plt.xlabel("num_mamba_blocks")
    plt.ylabel("Test Acc(%)")
    plt.title("Test Acc vs. #Blocks")
    plt.grid(True)

    # 3) F1 Score
    plt.subplot(1,3,3)
    plt.plot(df["num_mamba_blocks"], df["f1"], marker='o', color='green')
    plt.xlabel("num_mamba_blocks")
    plt.ylabel("F1")
    plt.title("F1 vs. #Blocks")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 예) root_dir = "multirun/2025-02-01/21-34-43"
    root_dir = "multirun/2025-02-01/21-34-43"
    df = collect_results(root_dir)
    print(df)

    if not df.empty:
        plot_metrics(df)
    else:
        print("No results found or no logs parsed.")

"""
python collect_and_plot_log.py

"""