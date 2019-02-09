import numpy as np

class DT:
	"""docstring for DT"""
	def __init__(self, depth=0, hist_dim=5):
		self.depth = depth # Глубина дерева
		self.hist_dim = hist_dim

	# Для решения задачи классификации (бинарная и мульти)
	def count_gini(self, y):
		'''
			
			(от k=1 до K)∑ i-катое * (1 - i-катое)
			где i-катое это доля уникального объекта в векторе

		'''
		lenght = len(y)
		unique = set(y)
		p = [len(list(filter(lambda x: x == i, y)))/lenght for i in unique]
		result = sum([i*(1-i) for i in p])
		return result


	def split_node(self, X, j, t):
		mask = X[:,j] < t
		return mask, ~mask


	def count_Q(self, X, y, j, t):
		H_x = self.count_gini(y)
		mask_l, mask_r = self.split_node(X, j, t)
		y_l, y_r = y[mask_l], y[mask_r]
		H_l, H_r = self.count_gini(y_l), self.count_gini(y_r)
		n_l, n_r, n = len(y_l), len(y_r), len(y)
		return H_x - ((n_l/n)*H_l) - ((n_r/n)*H_r)

	
	def count_hist(self, feature_vect):
		'''
			Перцентильное разбиение вектора фичи.
			Для оптимального поиска порогов разбиения.
			Пока только для вещественных признаков
		'''
		quant_vect = np.arange(1, self.hist_dim)*(100/self.hist_dim)
		return np.percentile(feature_vect, quant_vect)


	def fit(self, X, y):
		result = []
		for j in range(X.shape[1]):
			thresholds = self.count_hist(X[:,j])
			q = [self.count_Q(X, y, j, t) for t in thresholds]
			best_t_index = np.argmin(q)
			result.append(thresholds[best_t_index])
		print(np.argmin(result), result.min())


	def predict(self, X):
		...



if __name__ == '__main__':
	X = np.array([[1,2,3,4,5,6]]).T
	y = np.array([0,0,0,0,1,1])
	test_1 = ([0,1], 0.5)
	t = DT()
	test_1_r = t.count_gini(test_1[0])
	expected = test_1[1]
	# print('Test_1:', test_1_r == expected)
	# print(t.split_node(X, j=0, t=1.5))
	
	# print(t.count_Q(X, y, j=0, t=1.5))

	# print(t.count_hist(np.arange(1,100)))

	print(t.fit(X, y))

