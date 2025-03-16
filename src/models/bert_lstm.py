from transformers import  BertModel, BertConfig
import torch.nn as nn
import torch

class Lstm(nn.Module):
    def __init__(self, config):
        super(Lstm, self).__init__()
        rnn = nn.LSTM
        lstm_hidden_size = 200

        self.rnn1 = rnn(768, int(lstm_hidden_size), bidirectional=True)
        self.rnn2 = rnn(int(lstm_hidden_size), int(lstm_hidden_size), bidirectional=True)

        self.bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=self.bertconfig)

        self.fc = nn.Linear(800, 3)

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
        o = self.fc(o)

        return o
    
if __name__ == "__main__":
    from parameter_counter import count_parameters
    from omegaconf import DictConfig

    # 예시 Config 정의
    config = DictConfig({
    })

    # 모델 생성
    model = Lstm(config)

    # 파라미터 수 계산
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")  # 총 파라미터 수를 출력