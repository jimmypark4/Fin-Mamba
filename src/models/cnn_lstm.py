import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class CnnLstm(nn.Module):
    def __init__(self, config):
        super(CnnLstm, self).__init__()

        # CNN Layer
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # LSTM Layer
        lstm_hidden_size = 200
        self.rnn1 = nn.LSTM(128, lstm_hidden_size, bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(lstm_hidden_size * 2, lstm_hidden_size, bidirectional=True, batch_first=True)

        # BERT
        self.bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=self.bertconfig)

        # Fully connected layer for output
        self.fc = nn.Linear(lstm_hidden_size * 4, 3)

    def forward(self, bert_sent, bert_sent_type, bert_sent_mask):
        # BERT 모델 출력
        bert_output = self.bertmodel(input_ids=bert_sent,
                                     attention_mask=bert_sent_mask,
                                     token_type_ids=bert_sent_type)
        bert_output = bert_output[0]  # [B, seq_len, 768]
        batch_size = bert_output.shape[0]

        # CNN 처리: 입력 shape 변경 및 CNN 적용
        bert_output = bert_output.permute(0, 2, 1)  # [B, 768, seq_len]
        x = self.conv1(bert_output)  # [B, 256, seq_len]
        x = self.conv2(x)            # [B, 128, seq_len]
        x = self.pool(x)             # [B, 128, seq_len/2]
        x = x.permute(0, 2, 1)       # [B, reduced_seq_len, 128]

        # LSTM 처리
        # rnn1: 전체 시퀀스 출력 사용 (출력 shape: [B, seq_len, hidden_size*2] = [B, L, 400])
        output1, (h1, _) = self.rnn1(x)
        # rnn2: 첫 LSTM의 전체 시퀀스 출력을 사용
        output2, (h2, _) = self.rnn2(output1)

        # 결합 후 완전 연결층
        # h1: shape (2, B, 200), h2: shape (2, B, 200)
        # 또는 output2의 마지막 타임스텝을 사용하는 방법도 고려할 수 있음.
        o = torch.cat((h1, h2), dim=2).transpose(0, 1).reshape(batch_size, -1)
        output = self.fc(o)

        return output
