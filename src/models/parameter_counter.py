# 파라미터 수를 계산하는 함수
def count_parameters(model):
    """
    PyTorch 모델의 총 파라미터 수를 계산합니다.

    Args:
        model (torch.nn.Module): PyTorch 모델

    Returns:
        int: 모델의 총 파라미터 수
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
