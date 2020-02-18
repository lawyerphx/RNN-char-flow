import numpy as np

a = np.arange(3)
b = np.arange(5)
a = a[:, np.newaxis]
print(a)
print(b)
print(a.shape)
print(b.shape)
print(a+b)
