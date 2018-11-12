def prob():
	res = 0
	for i in range(1, 7):
		res += (i/2**i)
	return (1/6) * res

if __name__ == '__main__':
	print(prob())
