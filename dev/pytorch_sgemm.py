import torch
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# 创建两个随机大矩阵
L = 5120
matrix1 = torch.randn(L, L,dtype=torch.float32)
matrix2 = torch.randn(L, L,dtype=torch.float32)

matrix1 = matrix1.to(device)
matrix2 = matrix2.to(device)

# 使用 torch.matmul() 进行矩阵乘法
result1 = torch.matmul(matrix1, matrix2)

result1.to("cpu")


# 检查两种方法得到的结果是否相同
#assert torch.allclose(result1, result2)