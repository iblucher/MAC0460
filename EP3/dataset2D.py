import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

'''
Retorna duas tuplas com amostras de uma distribuição normal 
multivariada.
'''
def plot2D(mean1, mean2, cov1, cov2, size1, size2):
	data1 = np.random.multivariate_normal(mean1, cov1, size1).T
	data2 = np.random.multivariate_normal(mean2, cov2, size2).T

	plt.plot(data1[0], data1[1], 'x', marker = '.')
	plt.plot(data2[0], data2[1], 'x', marker = '.')

	plt.axis('equal')
	plt.show()

	return (data1, data2)

mean1 = (4, 2)
mean2 = (10, 2)
cov1 = [[2, 0], [0, 2]]
cov2 = [[1.5, 3], [3, 1.5]]

#circulos
plot2D(mean1, mean2, cov1, cov1, 10000, 10000)

#um circulo menor que o outro
plot2D(mean1, mean2, cov1, cov1, 1000, 10000)

#achatados
plot2D(mean1, mean2, cov2, cov2, 10000, 10000)

#um achatado menor que o outro
plot2D(mean1, mean2, cov2, cov2, 1000, 10000)
