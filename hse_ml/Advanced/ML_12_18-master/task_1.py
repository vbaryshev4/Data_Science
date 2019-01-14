x = [1, 2, 3, 4, 5, 6, 7]
y = [100, 102, 103, 50, 55, 61, 70]

def mean(lst):
	if len(lst) > 0:
		return sum(lst)/len(lst)
	return 0


def mse(lst, mean_value):
	if len(lst) > 0:
		return sum([(i - mean_value)**2 for i in lst]) / len(lst)
	return 0


if __name__ == "__main__":
	pairs = (list(zip(x,y)))
	results = []
	for threshold in range(min(x)-1, max(x)+2):
		left = [i for i in pairs if i[0] >= threshold]
		right = [i for i in pairs if i[0] <= threshold]
		y_mean_left = mean([i[1] for i in left])
		y_mean_right = mean([i[1] for i in right])
		result = mse([i[1] for i in left], y_mean_left) \
			+ mse([i[1] for i in right], y_mean_right)
		results.append((result, threshold))
	print(min(results))

	# Дописать для второго уровня




