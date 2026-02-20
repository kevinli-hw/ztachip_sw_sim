import numpy as np

weights_arr = np.loadtxt("./weights_new_step3.txt").astype(np.int8)
weights_arr = np.asarray(weights_arr).reshape(-1)
print(weights_arr.shape)

i = 0
pcore_id = 0
nthread = 8
thread_id = 0
lane_id = 0
VECTOR_WIDTH = 8
top_id = i + (pcore_id * nthread + thread_id) * VECTOR_WIDTH + lane_id

num_chunks = 19
chunk = 0
w8 = []
for i in range(VECTOR_WIDTH):
    w8.append(weights_arr[nthread*8*i+2])
print(w8)