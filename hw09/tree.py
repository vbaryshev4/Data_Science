import numpy as np

class DT:
	"""docstring for DT"""
	def __init__(self, depth=10, hist_dim=5):
		self.depth = depth # Глубина дерева
		self.hist_dim = hist_dim # Количество бинов для квантилизации(разбития) вектора
		self.tree = dict()

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


	def fit_node(self, X, y):
		result = []
		for j in range(X.shape[1]):
			thresholds = self.count_hist(X[:,j])
			q = [self.count_Q(X, y, j, t) for t in thresholds]
			best_t_index = np.argmin(q)
			result.append((q[best_t_index], thresholds[best_t_index], j))
		return min(result)


	def __fit(self, X, y, level=0):
		node = {
			'feature': None,
			'threshold': None,
			'is_terminal': False,
			'l_subtree': None,
			'r_subtree': None,
			'answer': None,
			'level': level
		}
		if len(np.unique(y)) != 1 and self.depth != node['level']:
			_, threshold, feature_index = self.fit_node(X, y)
			print('Debug', 'threshold', threshold, 'index', feature_index)
			node['feature'] = feature_index
			node['threshold'] = threshold
			l_mask, r_mask = self.split_node(X, feature_index, threshold)
			node['l_subtree'] = self.__fit(X[l_mask], y[l_mask], node['level']+1)
			node['r_subtree'] = self.__fit(X[r_mask], y[r_mask], node['level']+1)
		else:
			node['is_terminal'] = True
			node['answer'] = np.argmax(np.bincount(y))

		return node

	def fit(self, X_train, y_train):
		self.tree = self.__fit(X_train, y_train)
		print('Tree is fitted')

	def __predict_object(self, x, tree):
		if tree['is_terminal']:
			return tree['answer']
		else:
			if x[tree['feature']] < tree['threshold']:
				return self.__predict_object(x, tree['l_subtree'])
			else:
				return self.__predict_object(x, tree['r_subtree'])


	def predict(self, X_test):
		r = [self.__predict_object(x, self.tree) for x in X_test]
		print(r)
		return r


if __name__ == '__main__':
	X_train = np.array([[1,2,3,4,5,6], [1,2,3,4,5,6]]).T
	y_train = np.array([0,0,0,0,1,1])
	test_1 = ([0,1], 0.5)
	t = DT()
	test_1_r = t.count_gini(test_1[0])
	expected = test_1[1]
	# print('Test_1:', test_1_r == expected)
	# print(t.split_node(X, j=0, t=1.5))
	
	# print(t.count_Q(X, y, j=0, t=1.5))

	# print(t.count_hist(np.arange(1,100)))

	t.fit(X_train, y_train)
	t.predict(X_train)

