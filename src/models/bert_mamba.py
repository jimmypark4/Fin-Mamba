from transformers import  BertModel, BertConfig
from mamba_ssm import Mamba
import torch.nn as nn
import torch

class LstmMamba(nn.Module):
    def __init__(self, config):
        """
        config: Hydra에서 넘어온 DictConfig (cfg.model)
        num_heads, num_layers: 추가 하이퍼파라미터
        """

        super(LstmMamba, self).__init__()
        rnn = nn.LSTM
        lstm_hidden_size = 200

        # 1) Bi-LSTM
        self.rnn1 = rnn(768, lstm_hidden_size, bidirectional=True)
        self.rnn2 = rnn(lstm_hidden_size, lstm_hidden_size, bidirectional=True)

        # # 2) BERT
        # # embedding_model = "bert-base-uncased"  # 하드코딩 or config에서 가져옴
        self.bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=self.bertconfig)

        # 3) Mamba 블록
        self.mamba1 = Mamba(
            d_model=800,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        self.mamba2 = Mamba(
            d_model=800,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        # 4) 최종 분류 레이어
        self.fc = nn.Linear(800, 3)

    def forward(self, bert_sent, bert_sent_type, bert_sent_mask):
        # A. BERT 임베딩
        bert_output = self.bertmodel(
            input_ids=bert_sent,
            attention_mask=bert_sent_mask,
            token_type_ids=bert_sent_type
        )
        bert_output = bert_output[0]  # (batch_size, seq_len, 768)

        # 마스크 평균 풀링
        masked_output = bert_sent_mask.unsqueeze(2) * bert_output
        mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)
        bert_output = torch.sum(masked_output, dim=1) / mask_len  # (batch_size, 768)

        # LSTM 입력: (seq_len=1, batch_size, 768)
        utterance_text = bert_output.unsqueeze(0)

        # B. Bi-LSTM 2개
        _, (h1, _) = self.rnn1(utterance_text)
        _, (h2, _) = self.rnn2(h1)
        # h1/h2 shape: (2, batch_size, 200)

        # (batch_size, 800)로 reshape
        o = torch.cat([h1, h2], dim=2).permute(1, 0, 2).contiguous().view(h1.shape[1], -1)
        # (batch_size, 2, 400) => (batch_size, 800)

        # (batch_size, 1, 800)
        o = o.unsqueeze(1)

        # C. Mamba 블록 2개
        out1 = self.mamba1(o)
        out2 = self.mamba2(out1)
        out2 = out2.squeeze(1)  # (batch_size, 800)

        # D. 최종 분류
        logits = self.fc(out2)  # (batch_size, 3)
        return logits
# 모델 초기화 및 파라미터 수 계산
if __name__ == "__main__":
    from parameter_counter import count_parameters
    from omegaconf import DictConfig

    # 예시 Config 정의
    config = DictConfig({
    })

    # 모델 생성
    model = LstmMamba(config)

    # 파라미터 수 계산
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")  # 총 파라미터 수를 출력