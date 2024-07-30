import numpy as np
np.random.seed(10)

l=5120
size = l*l
M = l
N = l
K = l
arr1 = np.random.rand(size).reshape(l,l)
arr2 = np.random.rand(size).reshape(l,l)
arr3 = np.random.rand(size).reshape(l,l)

# for m in range(M):
#     for n in range(N):
#         for k in range(K):
#             arr3[m,n] +=arr1[m][k]*arr2[k][n]
            
            


result = np.dot(arr1,arr2)
