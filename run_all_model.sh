#!/bin/bash

# 모델 이름 리스트
# models=(  "bert_lstm" "bert_lstm_transformer"  "bert_mamba" "mamba_lstm" "mamba_lstm_3" )
models=("finbert"  "finbert_abligation" "finbert_lora" "mamba_peft" "finbert_mamba3" )

# 각 모델 실행
for model in "${models[@]}"
do
  echo "Running model: $model"
  python main.py model=$model
done
