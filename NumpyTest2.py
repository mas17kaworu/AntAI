import numpy as np

import shutil
import datetime

now = datetime.datetime.now()

num = 1
thread_num = 1
timeString = str(now.isoformat())

#复制单个文件
shutil.copy("./ant_log_W_"+str(thread_num) + "/0.replay", "./replay_saved/" + timeString.replace(':', '-')+".txt")
#复制并重命名新文件
#shutil.copy("C:\\a\\2.txt","C:\\b\\121.txt")
#复制整个目录(备份)
#shutil.copytree("C:\\a","C:\\b\\new_a")

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


