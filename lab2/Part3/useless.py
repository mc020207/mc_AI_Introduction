import torch

# 创建两个张量
tensor1 = torch.tensor([[1], [4]])
tensor2 = torch.tensor([[7, 8, 9], [10, 11, 12]])

# 按列合并两个张量
merged = torch.cat([tensor1, tensor2], dim=1)

print(merged.shape)
print(merged)

print(merged[[0, 1], [2, 3]])
