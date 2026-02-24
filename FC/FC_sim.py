import numpy as np
import math
from ctypes import c_float

# ZTA PARAM
DATA_BIT_WIDTH = 16
SPU_DATA_BIT_WIDTH = 12

def f32(x):
    return c_float(x).value   # 强制舍入到 IEEE754 float32

def PopulateConvolutionQuantizationParams():
    # OP PARAM
    input_scale = np.float32(0.00392127875238657)
    output_scale = np.float32(0.01837160810828209)
    filter_scale = np.float32(0.0015383206773549318)

    multiplier = np.float32(input_scale * filter_scale / output_scale)
    q, shift = math.frexp(multiplier)
    q_fixed = int(round(q * (1<<31)))
    if q_fixed == 1 << 31:
        q_fixed /= 2
        shift += 1
    print(multiplier, q, q_fixed, shift)

def gen_weights():
    # FC Weights
    weights = np.load("./single_fc_output_MatMul.npy")  # 自动还原 dtype/shape
    print("FC weights shape: ", weights.shape)

    # ZTA PARAM
    NUM_PCORE = 4 # 4 for small, 8 for large
    NUM_THREAD_PER_CORE = 16
    VECTOR_WIDTH = 8
    IP_CHUNK_SIZE = 8 # defined in kernels/conv.h

    # OP PARAM
    topcnt = 10
    botcnt = 147

    # find nthread with min_extra
    nthread = -1
    min_extra = -1
    for i in range(int(NUM_THREAD_PER_CORE/2), NUM_THREAD_PER_CORE):
        dx = NUM_PCORE * i * VECTOR_WIDTH
        extra = int(dx * (topcnt + dx - 1)/dx - topcnt)
        if min_extra < 0 or extra < min_extra:
            min_extra = extra
            nthread = i

    # new shape
    dx = NUM_PCORE * nthread * VECTOR_WIDTH
    print("nthread, dx: ", nthread, dx)
    topcnt_new = (int((topcnt + dx - 1) / dx)) * dx
    botcnt_new = (int((botcnt + IP_CHUNK_SIZE -1)/IP_CHUNK_SIZE))*IP_CHUNK_SIZE
    # print(nthread, dx, topcnt_new, botcnt_new)
    shape_new = topcnt_new * botcnt_new

    weights_new = np.zeros((shape_new), dtype=np.int8)
    # print("New weights shape: ", weights_new.shape)

    # step 1
    for i in range(topcnt):
        for j in range(botcnt):
            weights_new[j+i*botcnt_new] = weights[i][j]
    arr_tmp = np.array(weights_new).reshape(topcnt_new, botcnt_new)
    np.savetxt("./weights_new_step1.txt", arr_tmp, fmt="%d", delimiter=" ")

    # step 2
    temp1 = np.zeros((shape_new), dtype=np.int8)
    temp2 = np.zeros((shape_new), dtype=np.int8)

    # transpose for each VECTOR_WIDTH*IP_CHUNK_SIZE(8x8) block
    for pid in range(0, 2):
        for i in range(0 if pid == 0 else dx, topcnt, 2*dx):
            for j in range(0, botcnt, IP_CHUNK_SIZE):
                for k in range(i, i+dx, VECTOR_WIDTH):
                    for l in range(0, VECTOR_WIDTH):
                        # copy IP_CHUNK_SIZE
                        for m in range(0, IP_CHUNK_SIZE):
                            temp1[l*IP_CHUNK_SIZE+m] = weights_new[(k+l)*botcnt_new+j+m]
                    # transpose
                    for kk in range(0, VECTOR_WIDTH):
                        for jj in range(0, IP_CHUNK_SIZE):
                            temp2[jj*VECTOR_WIDTH+kk] = temp1[kk*IP_CHUNK_SIZE+jj]
                    for l in range(0, VECTOR_WIDTH):
                        # copy IP_CHUNK_SIZE
                        for m in range(0, IP_CHUNK_SIZE):
                            weights_new[(k + l) * botcnt_new + j + m] = temp2[l*IP_CHUNK_SIZE+m]

    arr_tmp = np.array(weights_new).reshape(topcnt_new, botcnt_new)
    np.savetxt("./weights_new_step2.txt", arr_tmp, fmt="%d", delimiter=" ")

    # step 3
    # final shape (topcnt_new, botcnt_new/IP_CHUNK_SIZE, IP_CHUNK_SIZE)
    h = topcnt_new
    w = int(botcnt_new/IP_CHUNK_SIZE)
    temp3 = np.zeros((h*w*IP_CHUNK_SIZE), dtype=np.int8)
    # transpose from (h, w, IP_CHUNK_SIZE) to (w, h, IP_CHUNK_SIZE)
    arr_tmp = np.array(weights_new).reshape(h, w, IP_CHUNK_SIZE)
    np.savetxt("./before_step3.txt", arr_tmp[0], fmt="%d", delimiter=" ")

    for r1 in range(0, h):
        for r2 in range(0, w):
            for m in range(0, IP_CHUNK_SIZE):
                temp3[r2*h*IP_CHUNK_SIZE+r1*IP_CHUNK_SIZE+m] = weights_new[r1*w*IP_CHUNK_SIZE+r2*IP_CHUNK_SIZE+m]

    arr_tmp = np.array(temp3).reshape(h, w, IP_CHUNK_SIZE)
    np.savetxt("./after_step3.txt", arr_tmp[0], fmt="%d", delimiter=" ")

    # copy final weights
    for r1 in range(0, h):
        for r2 in range(0, w):
            for r3 in range(0, IP_CHUNK_SIZE):
                weights_new[r3+r2*IP_CHUNK_SIZE+r1*w*IP_CHUNK_SIZE] = temp3[r3+r2*IP_CHUNK_SIZE+r1*w*IP_CHUNK_SIZE]

    arr_tmp = np.array(weights_new).reshape(topcnt_new, botcnt_new)
    np.savetxt("./weights_new_step3.txt", arr_tmp, fmt="%d", delimiter=" ")

def gen_bias():
    # OP PARAM
    output_shift = -11
    output_activation_min = -128
    output_offset = 1
    output_multiplier = 1444066619

    # what is D??
    D = 1 << (31-output_shift)
    bias = int((output_activation_min - output_offset)*D/output_multiplier)
    bias_len = 10
    bias_data = np.load("./single_fc_output_BiasAdd_ReadVariableOp.npy")  # 自动还原 dtype/shape
    # print("FC bias shape: ", bias_data.shape)
    bias_hi = np.zeros((10), dtype=np.int16)
    bias_lo = np.zeros((10), dtype=np.int16)
    # bias_range = 1 << (DATA_BIT_WIDTH - 2)
    bias_range = 1 << (DATA_BIT_WIDTH - 1)
    for i in range(bias_len):
        v = bias_data[i] - bias
        hi = np.int16(v/bias_range)
        lo = np.int16(v%bias_range)
        bias_hi[i] = hi
        bias_lo[i] = lo
    print("bias_hi: ", bias_hi)
    print("bias_lo: ", bias_lo)
    biasHi_array = np.array(bias_hi)
    biasLo_array = np.array(bias_lo)
    np.savetxt("./biasHi.txt", biasHi_array, fmt="%d", delimiter=" ")
    np.savetxt("./biasLo.txt", biasLo_array, fmt="%d", delimiter=" ")
    return

def SpuEvalActivation():
    return

def inner_product():
    # ZTA PARAM
    NUM_PCORE = 4 # 4 for small, 8 for large
    VECTOR_WIDTH = 8

    # FCN PARAM
    coef = np.loadtxt("./weights_new_step3.txt", delimiter=" ")
    biasHi = np.loadtxt("./biasHi.txt")
    biasLo = np.loadtxt("./biasLo.txt")
    # bot = 
    # input
    input_arr = [0 for i in range(3*7*7)]
    for i in range(3):
        for j in range(7):
            for k in range(7):
                input_arr[i*7*7+j*7+k] = i+j+k
    print(input_arr)

    # top = 
    topcnt = 10
    topdim = 1
    botcnt = 147
    botdim = 1
    coeftopcnt = 256
    coefbotcnt = 152

    # offset
    input_offset = -128
    filter_offset = 0
    output_offset = 1
    
    # top_scale=
    # stream =  # SpuBundle function
    num_thread = 8
    num_pcore = NUM_PCORE
    dx=num_pcore*num_thread*VECTOR_WIDTH
    
    return

def spu_function():
    return

import struct

def FLOAT2INT(in_val: float, data_bit_width: int = SPU_DATA_BIT_WIDTH) -> int:
    pos = data_bit_width - 1

    # 对应：v = *((unsigned int *)&in);
    # 把 Python float 按 IEEE754 单精度 float 打包，再按无符号 32 位整数解包
    v, = struct.unpack('>I', struct.pack('>f', in_val))

    if v == 0:
        result = 0
    else:
        flag = False

        # e = (v >> 23) & 0xff;
        e = (v >> 23) & 0xFF
        e2 = e - 127

        # v2 = (v & 0x007FFFFF);
        # v2 |= 0x00800000;
        v2 = v & 0x007FFFFF
        v2 |= 0x00800000

        # if (pos >= (e2 + 1)) { ... } else { ... }
        if pos >= (e2 + 1):
            shift = pos - e2 - 1
            if shift > 0:
                v2 >>= shift
        else:
            v2 = 0x00FFFFFF
            flag = True

        # result = (v2 >> 9) & (0x7fff);
        result = (v2 >> 9) & 0x7FFF

        # 四舍五入
        if v2 & 0x100:
            if result < 0x7FFF:
                result += 1
            else:
                flag = True

        # 符号
        if v & 0x80000000:
            result = -result

        # 饱和负向溢出：result = (short)0x8000 ==> -32768
        if flag and result < 0:
            result = -0x8000

    # C 里是 short 再右移 4 位，Python 的右移是算术右移，效果一致
    return result >> 4

def spu_eval_activation_scalar(_in_i16, SCALE, N, D, OFFSET, X_min, x_min, x_max):
    # 完全按 SW 的类型语义：float + int64常量会先转 float32
    x = f32(_in_i16)                  # (float)_in
    x = f32(x * f32(1 << SCALE))      # * (float)(1<<SCALE)
    x = f32(x + f32(X_min))           # + X_min (int64->float32)

    # 这里也要：N/D 先转 float32 再参与运算（C 里 int64 会转 float）
    xn = f32(x * f32(N))              # x*N
    rnd = f32(D/2) if xn > f32(0.0) else f32(-D/2)
    xq = f32((f32(xn + rnd) / f32(D)) + f32(OFFSET))

    # clamp
    if xq < f32(x_min): xq = f32(x_min)
    if xq > f32(x_max): xq = f32(x_max)

    return FLOAT2INT(xq)

#PopulateConvolutionQuantizationParams()
gen_weights()
gen_bias()
# inner_product()

# input_tensor = np.zeros((1,3,7,7), dtype=np.int8)
# for i in range(len(input_tensor[0])):
#     for j in range(len(input_tensor[0][i])):
#         for k in range(len(input_tensor[0][i][j])):
#             input_tensor[0][i][j][k] = i+j+k
# print(input_tensor)
# model_input = np.transpose(input_tensor, (0, 2, 3, 1)) # shape (1,7,7,3)
# model_input = model_input.reshape(-1)
# print(model_input)

input_tensor = np.zeros((1,3,7,7), dtype=np.int8)
for i in range(len(input_tensor[0])):
    for j in range(len(input_tensor[0][i])):
        for k in range(len(input_tensor[0][i][j])):
            input_tensor[0][i][j][k] = i+j+k
# model_input = np.transpose(input_tensor, (0, 2, 3, 1)) # shape (1,7,7,3)
model_input_i32 = input_tensor.reshape(-1).astype(np.int32)
bot_i32 = model_input_i32 + 128

# bias[0]: 3773, hi: 387, lo: 367
# bias[1]: 9058, hi: 392, lo: 532
# bias[2]: -12903, hi: 371, lo: 75
# bias[3]: 12831, hi: 396, lo: 209
# bias[4]: 10811, hi: 394, lo: 237
# bias[5]: 17618, hi: 400, lo: 900
# bias[6]: -6200, hi: 377, lo: 634
# bias[7]: -16930, hi: 367, lo: 144
# bias[8]: 4166, hi: 387, lo: 760
# bias[9]: -13741, hi: 370, lo: 261

# bias - X_min
bias_high = np.loadtxt("./biasHi.txt").astype(np.int16)
bias_low = np.loadtxt("./biasLo.txt").astype(np.int16)
bias_i32 = np.zeros((10), dtype=np.int32)
temp = 1 << (DATA_BIT_WIDTH - 1)
for i in range(10):
    bias_i32[i] = bias_high[i] * temp  + bias_low[i]
    temp_a = bias_high[i] * temp
    print(temp_a)

print(bias_i32)

weights = np.load("./single_fc_output_MatMul.npy").astype(np.int32)

result = [0 for i in range(10)]
result = np.asarray(result, dtype=np.int64)
final = np.zeros((10), dtype=np.int8)
SCALE = 9
N = 1444066619
D =4398046511104
OFFSET = 1
x_min=-128
x_max=127
X_min = -392882

for i in range(0, 10):
    # print(weights[i])
    result[i] = bias_i32[i] + np.sum(bot_i32 * weights[i, :], dtype=np.int64)

out_int12 = []
for i in range(10):
    _in_i16 = np.int16(result[i] >> SCALE)  # pcore: _A >> out_scale => vint16
    print(_in_i16)
    out_int12.append(spu_eval_activation_scalar(int(_in_i16), SCALE, N, D, OFFSET, X_min, x_min, x_max))

# out_int12 = spu_eval_activation_int_to_int12(
#     acc_i64=result,
#     SCALE=SCALE, N=N, D=D,
#     OFFSET=OFFSET,
#     X_min=X_min,
#     x_min=x_min, x_max=x_max
# )
print("final: ", out_int12)


