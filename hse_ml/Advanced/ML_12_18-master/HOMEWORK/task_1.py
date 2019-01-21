'''
	Постройте регрессионное дерево для прогнозирования y 
	с помощью x на обучающей выборке:
	Критерий деления узла на два – минимизация MSE. 
	Узлы делятся до тех пор, пока в узле остаётся больше двух наблюдений.
'''

def mean(lst):
	if len(lst) > 0:
		return sum(lst)/len(lst)
	return 0


def mse(lst, mean_value):
	if len(lst) > 0:
		return sum([(i - mean_value)**2 for i in lst]) / len(lst)
	return 0


if __name__ == "__main__":
	x = [1, 2, 3, 4, 5, 6, 7]
	y = [100, 102, 103, 50, 55, 61, 70]
	pairs = [(list(zip(x,y)))]
	tree_depth = 7

	def build_level(pairs):
		results = []
		X = [i[0] for i in pairs]
		for threshold in range(min(X)-1, max(X)+2):
			left = [i for i in pairs if i[0] >= threshold]
			right = [i for i in pairs if i[0] <= threshold]
			y_mean_left = mean([i[1] for i in left])
			y_mean_right = mean([i[1] for i in right])
			result = mse(
				[i[1] for i in left], y_mean_left
					) + mse(
				[i[1] for i in right], y_mean_right
					)
			results.append((result, threshold))
		return min(results)



	for level in range(0, tree_depth):
		cache = []
		for item in pairs:
			if len(item) > 2:
				t = build_level(item)
				r = 'Level={0} Node={1} Threshold={2} MSE={3}'.format(
					level, item, t[1], t[0])
				print(r)
				split_index = [i[0] for i in item].index(t[1]) + 1
				cache.append(item[:split_index])
				cache.append(item[split_index:])
		pairs = cache