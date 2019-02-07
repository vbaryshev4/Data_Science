import numpy as np

class DT:
	"""docstring for DT"""
	def __init__(self, depth=0):
		self.depth = depth # Глубина дерева

	def count_gini(self, y):
		lenght = len(y)
		unique = set(y)
		p = [len(list(filter(lambda x: x == i, y)))/lenght for i in unique]
		result = sum([i*(1-i) for i in p])
		return result

	def split_node(self, X, j, t):
		mask = X[:,j] < t
		print(X[:,j])
		print(mask, ~mask)
		return mask, ~mask

	def count_Q(self, X, y, j, t):
		H_x = self.count_gini(y)
		mask_l, mask_r = self.split_node(X, j, t)
		print(mask_l, mask_r)


class Node():
	def __init__(self, t, j):
		self.t = t # Порог разбиения
		self.j = j # Индекс фичи


if __name__ == '__main__':
	X = np.array([[1,2]]).T

	test_1 = ([0,1], 0.5)
	t = DT()
	test_1_r = t.count_gini(test_1[0])
	expected = test_1[1]
	print('Test_1:', test_1_r == expected)

	
	t.split_node(X, j=0, t=1.5)





		