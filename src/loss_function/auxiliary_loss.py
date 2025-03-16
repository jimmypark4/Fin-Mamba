# src/losses/auxiliary_loss.py
import torch
import torch.nn as nn

class LabelSmoothingLoss(nn.Module):
    """
    간단한 Label Smoothing 구현 예시
    - cross_entropy와 별개로 auxiliary term만 계산
    - Main loss와 더해 최종 loss로 사용 가능
    """
    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits, target):
        """
        logits: (batch_size, num_classes)
        target: (batch_size)  [0..num_classes-1]
        """
        num_classes = logits.size(1)
        with torch.no_grad():
            # One-hot label 생성 (batch_size, num_classes)
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            # correct label에 (1 - smoothing) 할당
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = torch.log_softmax(logits, dim=1)
        loss = -true_dist * log_probs  # (batch_size, num_classes)

        if self.reduction == 'mean':
            loss = loss.sum(dim=1).mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        else:
            # 'none' 등으로 반환
            loss = loss.sum(dim=1)
        return loss
