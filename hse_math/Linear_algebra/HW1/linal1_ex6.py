import numpy as np

def display_m(np_object, power):
	print(np_object * power, '\n')

if __name__ == '__main__':
    
	A = [
			[4,1,0],
			[0,4,1],
			[0,0,4]
		]
	B = [
			[2,1,0,0,0],
			[0,2,0,0,0],
			[0,0,-3,1,0],
			[0,0,0,-3,1],
			[0,0,0,0,3]
		]
	A = np.matrix(A)
	B = np.matrix(B)

for matrix in [A, B]:
	for power in [2,3,4]:
		display_m(matrix, power)