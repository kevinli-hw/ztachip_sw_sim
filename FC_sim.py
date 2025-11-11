import numpy as np
import math


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
    weights = np.load("single_fc_output_MatMul.npy")  # 自动还原 dtype/shape
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
    np.savetxt("weights_new_step1.txt", arr_tmp, fmt="%d", delimiter=" ")

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
    np.savetxt("weights_new_step2.txt", arr_tmp, fmt="%d", delimiter=" ")

    # step 3
    # final shape (topcnt_new, botcnt_new/IP_CHUNK_SIZE, IP_CHUNK_SIZE)
    h = topcnt_new
    w = int(botcnt_new/IP_CHUNK_SIZE)
    temp3 = np.zeros((h*w*IP_CHUNK_SIZE), dtype=np.int8)
    # transpose from (h, w, IP_CHUNK_SIZE) to (w, h, IP_CHUNK_SIZE)
    arr_tmp = np.array(weights_new).reshape(h, w, IP_CHUNK_SIZE)
    np.savetxt("before_step3.txt", arr_tmp[0], fmt="%d", delimiter=" ")

    for r1 in range(0, h):
        for r2 in range(0, w):
            for m in range(0, IP_CHUNK_SIZE):
                temp3[r2*h*IP_CHUNK_SIZE+r1*IP_CHUNK_SIZE+m] = weights_new[r1*w*IP_CHUNK_SIZE+r2*IP_CHUNK_SIZE+m]

    arr_tmp = np.array(temp3).reshape(h, w, IP_CHUNK_SIZE)
    np.savetxt("after_step3.txt", arr_tmp[0], fmt="%d", delimiter=" ")

    # copy final weights
    for r1 in range(0, h):
        for r2 in range(0, w):
            for r3 in range(0, IP_CHUNK_SIZE):
                weights_new[r3+r2*IP_CHUNK_SIZE+r1*w*IP_CHUNK_SIZE] = temp3[r3+r2*IP_CHUNK_SIZE+r1*w*IP_CHUNK_SIZE]

    arr_tmp = np.array(weights_new).reshape(topcnt_new, botcnt_new)
    np.savetxt("weights_new_step3.txt", arr_tmp, fmt="%d", delimiter=" ")

def gen_bias():
    # ZTA PARAM
    DATA_BIT_WIDTH = 12

    # OP PARAM
    output_shift = -11
    output_activation_min = -128
    output_offset = 1
    output_multiplier = 1444066619

    # what is D??
    D = 1 << (31-output_shift)
    bias = int((output_activation_min - output_offset)*D/output_multiplier)
    bias_len = 10
    bias_data = np.load("single_fc_output_BiasAdd_ReadVariableOp.npy")  # 自动还原 dtype/shape
    # print("FC bias shape: ", bias_data.shape)
    bias_hi = np.zeros((10), dtype=np.int16)
    bias_lo = np.zeros((10), dtype=np.int16)
    bias_range = 1 << (DATA_BIT_WIDTH - 2)
    for i in range(bias_len):
        v = bias_data[i] - bias
        hi = np.int16(v/bias_range)
        lo = np.int16(v%bias_range)
        bias_hi[i] = hi
        bias_lo[i] = lo
    # print("bias_hi: ", bias_hi)
    # print("bias_lo: ", bias_lo)
    biasHi_array = np.array(bias_hi)
    biasLo_array = np.array(bias_lo)
    np.savetxt("biasHi.txt", biasHi_array, fmt="%d", delimiter=" ")
    np.savetxt("biasLo.txt", biasLo_array, fmt="%d", delimiter=" ")
    return

def SpuEvalActivation():
    

def inner_product():
    # ZTA PARAM
    NUM_PCORE = 4 # 4 for small, 8 for large
    VECTOR_WIDTH = 8

    # FCN PARAM
    coef = np.loadtxt("./weights_new_step3.txt", delimiter=" ")
    biasHi = np.loadtxt("./biasHi.txt")
    biasLo = np.loadtxt("./biasLo.txt")
    # bot = 
    # top = 
    topcnt = 10
    topdim = 1
    botcnt = 147
    botdim = 1
    coeftopcnt = 256
    coefbotcnt = 152
    
    # top_scale=
    # stream =  # SpuBundle function
    num_thread = 8
    num_pcore = NUM_PCORE
    dx=num_pcore*num_thread*VECTOR_WIDTH
    
    return

def spu_function():
    return

#PopulateConvolutionQuantizationParams()
gen_weights()
#gen_bias()
# inner_product()