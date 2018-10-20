import numpy as np
from numpy.linalg import matrix_power


# Управжнение №1
def count(x):
	return x**3-6*x**2+11*x-6

for i in range(-10, 10):
	r = count(i) 
	print(i, r)

# Проверка решения
T = [[1,1,1],[1,0,2],[2,1,2]]
B = [[1,0,0],[0,2,0],[0,0,3]]
TT = [[-2,-1,2],[2,0,-1],[1,1,-1]] #T-1

X = np.matmul(np.matmul(T,B), TT)
print(X)


