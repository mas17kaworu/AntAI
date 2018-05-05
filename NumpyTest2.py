import numpy as np

# a = [[1, 2, 3], [4, 5, 6]]
# arr = np.array(a)
# print(arr)
# arr = arr.flatten()
# print(arr)
# loc = (5, 6)
# arr = np.insert(arr, arr.shape, loc)
# print(arr)

buffer_r = [1, 2, 3, 4]
GAMMA = 0.9
v_s_ = 1
buffer_v_target = []
for r in buffer_r[::-1]:    # reverse buffer r
    v_s_ = r + GAMMA * v_s_
    buffer_v_target.append(v_s_)
buffer_v_target.reverse()
print(buffer_v_target)
