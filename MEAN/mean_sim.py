import numpy as np
import math

input_scale = 0.023529
output_scale = 0.014821

multiplier = np.float32(input_scale / output_scale)
q, shift = math.frexp(multiplier)
q_fixed = int(round(q * (1<<31)))
if q_fixed == 1 << 31:
    q_fixed /= 2
    shift += 1
print(multiplier, q, q_fixed, shift)

real_scale = np.float32(input_scale / output_scale)
hw_scale = np.float32(q_fixed / (2**(31-shift)))
print(real_scale, hw_scale)

input_shift = -128
output_shift = -128

input_image = np.zeros((8,7,7), dtype=np.int8)

for i in range(len(input_image)):
    for j in range(len(input_image[i])):
        for k in range(len(input_image[i][j])):
            if (i+j+k) % 2 == 0:
                input_image[i][j][k] = -((i*j*k)%128)
            else:
                input_image[i][j][k] = ((i*j*k)%128)

input_image = np.transpose(input_image, (1, 2, 0)).astype(np.int16)
# print(input_image)
result_image = np.zeros((8), dtype=np.int32)
# # mean
# # add
input_image = input_image - input_shift
# print(input_image)
for i in range(len(input_image)):
    for j in range(len(input_image[i])):
        for k in range(len(input_image[i][j])):
            result_image[k] += input_image[i][j][k]

print(result_image)

result_image = (np.int8)((result_image * hw_scale / 49) + output_shift)
print(result_image)