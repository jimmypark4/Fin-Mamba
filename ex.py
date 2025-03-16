

# import torch
# from mamba_ssm import Mamba

# batch, length, dim = 2, 64, 16
# x = torch.randn(batch, length, dim).to("cuda")
# model = Mamba(
#     # This module uses roughly 3 * expand * d_model^2 parameters
#     d_model=dim, # Model dimension d_model
#     d_state=16,  # SSM state expansion factor
#     d_conv=4,    # Local convolution width
#     expand=2,    # Block expansion factor
# ).to("cuda")
# y = model(x)
# print(y)

# import torch
# from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer

# # GPU 메모리 사용량 확인 함수
# def print_gpu_memory_usage():
#     allocated = torch.cuda.memory_allocated()
#     reserved = torch.cuda.memory_reserved()
#     print(f"GPU Memory Allocated: {allocated / 1024**2:.2f} MB")
#     print(f"GPU Memory Reserved: {reserved / 1024**2:.2f} MB")

# # 모델 로드
# tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
# model = MambaForCausalLM.from_pretrained("state-spaces/mamba-2.8b-hf").to("cuda")

# # 입력 생성
# input_ids = tokenizer("Hey, how are you doing?", return_tensors="pt")["input_ids"].to("cuda")

# # 메모리 사용량 확인 (모델 로드 후)
# print("After model loading:")
# print_gpu_memory_usage()

# # 추론 수행
# with torch.no_grad():
#     output = model.generate(input_ids, max_new_tokens=10)

# # 메모리 사용량 확인 (추론 후)
# print("After inference:")
# print_gpu_memory_usage()

# # 결과 출력
# print("Generated text:", tokenizer.batch_decode(output, skip_special_tokens=True))




