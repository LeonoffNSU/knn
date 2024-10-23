import mglearn
import numpy as np
import matplotlib.pyplot as plt
from mglearn import datasets, discrete_scatter
from statistics import mode


X, y = datasets.make_forge()
discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(['Класс 0', 'Класс 1'], loc=4)
plt.xlabel('Первый признак')
plt.ylabel('Второй признак')
plt.show()


y = y[:, np.newaxis]
print(y)
q = np.array([10, 3])
print(X)
print(q)
X = np.hstack([X, y])
print(X)
#result[0] = sqrt((9.96 - 10) ^ 2 - (4.59 - 3) ^ 2)
result = []
for row in X:
    difference = row[:2] - q
    difference **= 2
    result.append([np.sqrt(np.sum(difference)), row[-1]])
    #print(difference)


#print(result)
result.sort(key=lambda x: x[0])
#distances = np.sort(result)
print(result)


def knn(k):
    knn = np.array(result[0:k])
    print(knn)
    classes = knn[:, 1]
    print(classes)
    return mode(classes)

print(knn(5))