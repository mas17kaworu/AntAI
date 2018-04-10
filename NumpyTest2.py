import numpy as np

a = [[1, 2, 3], [4, 5, 6]]
arr = np.array(a)
print(arr)
arr = arr.flatten()
print(arr)
loc = (5, 6)
arr = np.insert(arr, arr.shape, loc)
print(arr)
