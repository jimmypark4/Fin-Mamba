import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
import xgboost as xgb
import numpy as np

class LstmXgboost(nn.Module):
    def __init__(self, config):
        super(LstmXgboost, self).__init__()
        rnn = nn.LSTM
        lstm_hidden_size = 200

        self.rnn1 = rnn(768, int(lstm_hidden_size), bidirectional=True)
        self.rnn2 = rnn(int(lstm_hidden_size), int(lstm_hidden_size), bidirectional=True)

        self.bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=self.bertconfig)

        self.fc = nn.Linear(800, 3)
        
        # XGBoost 모델
        self.xgb_model = xgb.XGBClassifier(objective="multi:softmax", num_class=3)

    def forward(self, bert_sent, bert_sent_type, bert_sent_mask):
        bert_output = self.bertmodel(input_ids=bert_sent,
                                     attention_mask=bert_sent_mask,
                                     token_type_ids=bert_sent_type)

        bert_output = bert_output[0]

        batch_size = bert_output.shape[0]

        masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
        mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)
        bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len

        utterance_text = torch.unsqueeze(bert_output, axis=0)

        _, (h1, _) = self.rnn1(utterance_text)
        _, (h2, _) = self.rnn2(h1)

        o = torch.cat((h1, h2), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        lstm_output = self.fc(o)

        # XGBoost 학습을 위한 출력값 반환
        return lstm_output

    def train_xgb(self, X_train, y_train):
        """
        X_train: LSTM을 통해 나온 특성들을 사용한 데이터 (훈련)
        y_train: 훈련용 라벨
        """
        self.xgb_model.fit(X_train, y_train)
    
    def predict_xgb(self, X_test):
        """
        X_test: LSTM을 통해 나온 특성들을 사용한 데이터 (예측)
        """
        return self.xgb_model.predict(X_test)

if __name__ == "__main__":
    from parameter_counter import count_parameters
    from omegaconf import DictConfig

    # 예시 Config 정의
    config = DictConfig({})

    # 모델 생성
    model = LstmXgboost(config)

    # 파라미터 수 계산
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")  # 총 파라미터 수를 출력
