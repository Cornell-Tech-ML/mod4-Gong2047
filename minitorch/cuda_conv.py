# cuda_conv.py
import math
import numpy as np
from numba import cuda

# ============ CUDA Kernel Implementations ============

@cuda.jit
def conv1d_kernel(in_data, weight, out_data):
    b, ic, w_in = in_data.shape
    oc, ic2, kw = weight.shape

    # 当前线程负责的输出元素位置
    b_idx, oc_idx, w_idx = cuda.grid(3)

    if b_idx < b and oc_idx < oc and w_idx < (w_in - kw + 1):
        val = 0.0
        for ic_idx in range(ic):
            for k in range(kw):
                val += in_data[b_idx, ic_idx, w_idx + k] * weight[oc_idx, ic_idx, k]
        out_data[b_idx, oc_idx, w_idx] = val


def conv1d(in_data: np.ndarray, weight: np.ndarray) -> np.ndarray:
    B, IC, W = in_data.shape
    OC, IC2, KW = weight.shape
    assert IC == IC2

    W_out = W - KW + 1
    out_data = np.zeros((B, OC, W_out), dtype=in_data.dtype)

    threadsperblock = (8, 8, 8)
    bpg_b = math.ceil(B / threadsperblock[0])
    bpg_oc = math.ceil(OC / threadsperblock[1])
    bpg_w = math.ceil(W_out / threadsperblock[2])
    blockspergrid = (bpg_b, bpg_oc, bpg_w)

    d_in = cuda.to_device(in_data)
    d_weight = cuda.to_device(weight)
    d_out = cuda.to_device(out_data)

    conv1d_kernel[blockspergrid, threadsperblock](d_in, d_weight, d_out)
    d_out.copy_to_host(out_data)
    return out_data


@cuda.jit
def conv2d_kernel(in_data, weight, out_data):
    b, ic, h_in, w_in = in_data.shape
    oc, ic2, kh, kw = weight.shape

    # 当前线程负责的输出元素位置
    b_idx, oc_idx, hw_idx = cuda.grid(3)

    H_out = h_in - kh + 1
    W_out = w_in - kw + 1

    if b_idx < b and oc_idx < oc and hw_idx < (H_out * W_out):
        h_idx = hw_idx // W_out
        w_idx = hw_idx % W_out
        val = 0.0
        for ic_idx in range(ic):
            for kh_idx in range(kh):
                for kw_idx in range(kw):
                    val += in_data[b_idx, ic_idx, h_idx + kh_idx, w_idx + kw_idx] * weight[oc_idx, ic_idx, kh_idx, kw_idx]
        out_data[b_idx, oc_idx, h_idx, w_idx] = val


def conv2d(in_data: np.ndarray, weight: np.ndarray) -> np.ndarray:
    B, IC, H, W = in_data.shape
    OC, IC2, KH, KW = weight.shape
    assert IC == IC2

    H_out = H - KH + 1
    W_out = W - KW + 1
    out_data = np.zeros((B, OC, H_out, W_out), dtype=in_data.dtype)

    threadsperblock = (8, 8, 8)
    bpg_b = math.ceil(B / threadsperblock[0])
    bpg_oc = math.ceil(OC / threadsperblock[1])
    bpg_hw = math.ceil((H_out * W_out) / threadsperblock[2])
    blockspergrid = (bpg_b, bpg_oc, bpg_hw)

    d_in = cuda.to_device(in_data)
    d_weight = cuda.to_device(weight)
    d_out = cuda.to_device(out_data)

    conv2d_kernel[blockspergrid, threadsperblock](d_in, d_weight, d_out)
    d_out.copy_to_host(out_data)
    return out_data

# ============ CPU reference implementations ============

def cpu_conv1d(in_data: np.ndarray, weight: np.ndarray) -> np.ndarray:
    B, IC, W = in_data.shape
    OC, IC2, KW = weight.shape
    assert IC == IC2
    W_out = W - KW + 1
    out_data = np.zeros((B, OC, W_out), dtype=in_data.dtype)

    for b in range(B):
        for oc in range(OC):
            for w in range(W_out):
                val = 0.0
                for ic in range(IC):
                    for k in range(KW):
                        val += in_data[b, ic, w+k] * weight[oc, ic, k]
                out_data[b, oc, w] = val
    return out_data

def cpu_conv2d(in_data: np.ndarray, weight: np.ndarray) -> np.ndarray:
    B, IC, H, W = in_data.shape
    OC, IC2, KH, KW = weight.shape
    assert IC == IC2
    H_out = H - KH + 1
    W_out = W - KW + 1
    out_data = np.zeros((B, OC, H_out, W_out), dtype=in_data.dtype)

    for b in range(B):
        for oc in range(OC):
            for h in range(H_out):
                for w in range(W_out):
                    val = 0.0
                    for ic in range(IC):
                        for kh in range(KH):
                            for kw in range(KW):
                                val += in_data[b, ic, h+kh, w+kw] * weight[oc, ic, kh, kw]
                    out_data[b, oc, h, w] = val
    return out_data

# ============ Test functions ============

def assert_allclose(cpu_result, gpu_result, atol=1e-5, rtol=1e-5):
    if not np.allclose(cpu_result, gpu_result, atol=atol, rtol=rtol):
        diff = np.abs(cpu_result - gpu_result)
        print("Max diff:", diff.max(), "at", np.unravel_index(diff.argmax(), diff.shape))
        raise AssertionError("CPU and GPU results do not match within tolerance.")
    else:
        print("Test passed! CPU and GPU results are close.")

# 非平凡测试1（1D）：随机输入与权重
def test_conv1d_random():
    np.random.seed(42)
    B, IC, W = 2, 3, 10
    OC, KW = 4, 3
    in_data = np.random.randn(B, IC, W).astype(np.float32)
    weight = np.random.randn(OC, IC, KW).astype(np.float32)

    cpu_res = cpu_conv1d(in_data, weight)
    gpu_res = conv1d(in_data, weight)

    print("1D random test CPU:\n", cpu_res)
    print("1D random test GPU:\n", gpu_res)
    assert_allclose(cpu_res, gpu_res)

# 非平凡测试2（1D）：有规律的输入与核
def test_conv1d_pattern():
    B, IC, W = 1, 2, 8
    OC, KW = 2, 3
    in_data = np.array([[[1,2,3,4,5,6,7,8],
                         [2,3,4,5,6,7,8,9]]], dtype=np.float32)
    weight = np.array([[[ -1, 0, 1],
                        [ -1, 0, 1]],
                       [[  1, 1, 1],
                        [  0, 0,-1]]], dtype=np.float32)

    cpu_res = cpu_conv1d(in_data, weight)
    gpu_res = conv1d(in_data, weight)

    print("1D pattern test CPU:\n", cpu_res)
    print("1D pattern test GPU:\n", gpu_res)
    assert_allclose(cpu_res, gpu_res)

# 非平凡测试1（2D）：随机输入与权重
def test_conv2d_random():
    np.random.seed(123)
    B, IC, H, W = 2, 3, 8, 8
    OC, KH, KW = 4, 3, 3
    in_data = np.random.randn(B, IC, H, W).astype(np.float32)
    weight = np.random.randn(OC, IC, KH, KW).astype(np.float32)

    cpu_res = cpu_conv2d(in_data, weight)
    gpu_res = conv2d(in_data, weight)

    print("2D random test CPU:\n", cpu_res)
    print("2D random test GPU:\n", gpu_res)
    assert_allclose(cpu_res, gpu_res)

# 非平凡测试2（2D）：有规律输入与Sobel风格卷积核
def test_conv2d_pattern():
    B, IC, H, W = 1, 1, 5, 5
    OC, KH, KW = 1, 3, 3
    # 简单的0-24递增输入
    in_data = np.arange(H*W).reshape(1,1,H,W).astype(np.float32)
    weight = np.array([[[[-1,0,1],
                         [-2,0,2],
                         [-1,0,1]]]]).astype(np.float32)

    cpu_res = cpu_conv2d(in_data, weight)
    gpu_res = conv2d(in_data, weight)

    print("2D pattern test CPU:\n", cpu_res)
    print("2D pattern test GPU:\n", gpu_res)
    assert_allclose(cpu_res, gpu_res)

if __name__ == "__main__":
    print("Running 1D conv tests...")
    test_conv1d_random()
    test_conv1d_pattern()

    print("\nRunning 2D conv tests...")
    test_conv2d_random()
    test_conv2d_pattern()


## Test log
## I ran it on my local machine since I have a GPU and everything's already set up with CUDA

# Running 1D conv tests...

# 1D random test CPU:
#  [[[-7.3251319e-01 -4.0113506e+00 -6.1490102e+00 -7.0634490e-01
#    -1.6243507e+00 -1.7813687e+00  9.8283306e-02 -4.6452479e+00]
#   [ 4.4455823e-01  8.3196430e+00  1.3832738e+00 -3.8361144e+00
#     4.2370863e+00 -6.0707289e-01  7.0955420e-01  2.7813292e+00]
#   [-1.8109510e+00 -1.6072683e+00 -4.6015491e+00 -7.5791532e-01
#    -1.4942508e+00 -4.3576312e+00 -5.1172185e-01  1.1716911e-01]
#   [-5.8754224e-01  4.0581756e+00  3.2937527e-04 -4.3928981e-01
#     2.6759465e+00 -1.9108260e+00  1.8240643e+00  1.4612592e+00]]

#  [[-1.5961809e+00 -8.5258317e-01 -1.7832861e+00  1.2147198e+00
#     7.1727067e-01  3.6137176e+00  3.4790783e+00 -1.7011325e+00]
#   [ 2.0167115e+00 -2.7832103e+00  3.4865372e+00 -1.0008283e+00
#     6.6219699e-01 -4.8304820e+00 -2.9311481e+00  5.6884823e+00]
#   [-3.8453195e+00  1.6863941e+00  3.3844731e+00 -3.5426412e+00
#    -9.6606457e-01 -2.1972921e+00  6.5625281e+00  5.1926680e+00]
#   [ 1.6836137e-01 -3.1438893e-01  1.8626821e-01 -4.3337479e+00
#     1.0321832e+00 -2.2996125e+00  1.0745405e+00  3.7733495e-01]]]
# 1D random test GPU:
#  [[[-7.3251319e-01 -4.0113506e+00 -6.1490097e+00 -7.0634490e-01
#    -1.6243507e+00 -1.7813686e+00  9.8283373e-02 -4.6452479e+00]
#   [ 4.4455826e-01  8.3196430e+00  1.3832738e+00 -3.8361142e+00
#     4.2370863e+00 -6.0707295e-01  7.0955414e-01  2.7813292e+00]
#   [-1.8109510e+00 -1.6072685e+00 -4.6015491e+00 -7.5791526e-01
#    -1.4942508e+00 -4.3576307e+00 -5.1172185e-01  1.1716917e-01]
#   [-5.8754230e-01  4.0581756e+00  3.2945164e-04 -4.3928981e-01
#     2.6759465e+00 -1.9108260e+00  1.8240641e+00  1.4612592e+00]]

#  [[-1.5961809e+00 -8.5258311e-01 -1.7832860e+00  1.2147199e+00
#     7.1727055e-01  3.6137176e+00  3.4790783e+00 -1.7011324e+00]
#   [ 2.0167117e+00 -2.7832100e+00  3.4865375e+00 -1.0008280e+00
#     6.6219699e-01 -4.8304815e+00 -2.9311481e+00  5.6884823e+00]
#   [-3.8453197e+00  1.6863941e+00  3.3844731e+00 -3.5426409e+00
#    -9.6606457e-01 -2.1972921e+00  6.5625281e+00  5.1926680e+00]
#   [ 1.6836141e-01 -3.1438893e-01  1.8626806e-01 -4.3337479e+00
#     1.0321832e+00 -2.2996125e+00  1.0745404e+00  3.7733486e-01]]]
# Test passed! CPU and GPU results are close.
# 1D pattern test CPU:
#  [[[ 4.  4.  4.  4.  4.  4.]
#   [ 2.  4.  6.  8. 10. 12.]]]
# 1D pattern test GPU:
#  [[[ 4.  4.  4.  4.  4.  4.]
#   [ 2.  4.  6.  8. 10. 12.]]]
# Test passed! CPU and GPU results are close.

# Running 2D conv tests...

# 2D random test CPU:
#  [[[[ -1.9538707    6.491252     9.357203     5.4295588    2.8569193
#      -4.9891415 ]
#    [ -2.0923505    3.289144    -6.705846     1.8944639    2.525137
#       0.3922958 ]
#    [ -6.0735545   -2.2464323   -6.049841     0.40801415   1.0304382
#      -2.461161  ]
#    [ -3.702654    -0.09293842  -2.5099788    1.2784364   -2.1177425
#       0.33235598]
#    [  6.029624     2.1655872   -3.3563       0.6458079    2.2073014
#      -0.8312354 ]
#    [  3.881553     6.0798664    2.509621     2.5929997   -0.9594182
#       1.0491652 ]]

#   [[ -5.2522426   -4.425784     3.7906039   -5.4776807    4.3491287
#      -1.2106485 ]
#    [ -6.125386    -1.0606674   -0.940258     2.264628     3.336758
#      10.916322  ]
#    [ -8.560209     2.847142    -1.3215171    0.8852517   -2.0804915
#      -3.5963378 ]
#    [  2.753674     2.2265363   -6.637757     0.3140323   -4.4109664
#     -12.98578   ]
#    [  3.3345437    8.183899    -5.452185     2.1671522   -2.60733
#       1.4117978 ]
#    [-10.743912    -0.12205482  -2.465022     1.8357713   -0.7580489
#       5.8631186 ]]

#   [[  5.7805357    5.502543    -6.1884027   -8.727333     2.037358
#       8.804388  ]
#    [ -0.67808163  -0.83318496   3.3737197    0.50331867   3.2287254
#       3.516023  ]
#    [ -0.37816462   7.750328     9.235642    -3.5693598    3.3006973
#       2.4809923 ]
#    [ -2.799384    -3.8555796    0.69558924   1.0687237    3.9249537
#      -8.181626  ]
#    [ -2.855411     7.002783    -1.9879277    4.967681     0.79862654
#      -2.574979  ]
#    [ -5.0481944   -1.5901065    3.2135632    5.7814975   -1.5875261
#       4.4493723 ]]

#   [[ 11.194473     2.3875394   -8.953257     5.496313     8.9386015
#      10.503974  ]
#    [  7.326025     7.5123057    1.9312749    1.2925929    5.2304454
#      -1.9248575 ]
#    [ -5.9565754   -0.77186835   3.174908     3.443634     0.73105454
#      -6.736043  ]
#    [ -8.251636     1.9667495    2.8968024    7.044662    13.394308
#      -9.774336  ]
#    [ -3.1389017    1.6156915    4.19819      0.21137705   6.751906
#       0.5825279 ]
#    [  2.1526852    3.7850258   -0.32172132   9.873247     1.0275081
#       3.2635682 ]]]


#  [[[  0.34238312   1.3348501   -7.337857     6.3890123   -2.6517735
#       0.8656528 ]
#    [ -4.8109646   -5.0405397   -0.70275974  -2.8767078   -1.8255703
#      -4.7250104 ]
#    [ -1.5542513   -4.87353      0.7409978    1.2531462   -0.51451814
#      -5.307119  ]
#    [  8.52045      3.9609582    3.3850124    3.0958889   -1.8593186
#       7.607734  ]
#    [  2.1711984    3.5862844   -0.9093183    4.139816    -1.7166018
#       3.0700867 ]
#    [ -0.52256024   9.289258    -3.0563173   -2.672453    -2.771598
#      -0.16351956]]

#   [[ -1.7723856   -7.713601     5.34827     -1.2680633   -4.252367
#      -2.374599  ]
#    [ -3.2018893    0.78604233   2.2069125    2.4339545    2.982395
#      -0.5857953 ]
#    [  6.2565217    0.9744816    4.4551024    7.4508314    3.4377804
#      -0.03961051]
#    [  5.0451984   -0.15618575  -1.3215286    9.75358    -14.767646
#       4.124437  ]
#    [  4.5069623   -1.9589522   -5.1757307    4.226739    -3.7731304
#       0.7927315 ]
#    [-11.193409     1.409735     2.7773666    2.3927362    1.3515432
#      -6.57348   ]]

#   [[ -2.9832573   -6.0503163   -2.6358998   -0.16181934   0.86787975
#      -3.3714    ]
#    [ 15.423568     3.3250933    7.5982404    3.0266385    4.909357
#      -3.8688645 ]
#    [  5.1565685    6.364912    -0.62687844   5.1238327   -3.2916288
#      -2.6995285 ]
#    [ -5.5202994   -2.4093537   -9.780078    -0.9002108    2.712024
#      -3.0801203 ]
#    [ -6.323711    -6.559602    -1.8065221   -2.2107856   -3.0023136
#      -3.7933416 ]
#    [ -3.9296625   -4.5197005    5.0114193    2.6360276   -4.446056
#       0.14866292]]

#   [[  1.5250856   -1.79179     -2.8472495    1.6935666   -2.9698014
#       2.21624   ]
#    [ 15.575943    11.604508    -2.3013225   -3.1636784    4.8202085
#      -2.6738646 ]
#    [ 12.817657     0.41151452  -7.722326    -1.1561267    1.5875459
#       3.2546215 ]
#    [ -9.323236    -5.665029   -13.759613     5.308601     3.4300992
#      -1.0695883 ]
#    [ -3.2727613   -2.927731    -9.0889845   -1.2734653    2.4471765
#      -3.8166523 ]
#    [-10.333294     1.5813686   -2.8698132    1.8504171    1.198906
#      -5.3794994 ]]]]
# 2D random test GPU:
#  [[[[ -1.9538702    6.4912515    9.357202     5.429559     2.8569198
#      -4.9891415 ]
#    [ -2.092351     3.2891438   -6.705846     1.8944639    2.5251367
#       0.39229593]
#    [ -6.0735555   -2.2464328   -6.0498405    0.40801397   1.0304385
#      -2.461161  ]
#    [ -3.7026532   -0.09293818  -2.5099788    1.2784358   -2.1177428
#       0.33235547]
#    [  6.0296235    2.1655874   -3.3563008    0.645808     2.2073011
#      -0.8312357 ]
#    [  3.881553     6.0798674    2.5096202    2.5929997   -0.95941824
#       1.0491656 ]]

#   [[ -5.252242    -4.425784     3.7906039   -5.47768      4.3491287
#      -1.2106484 ]
#    [ -6.125387    -1.0606672   -0.94025797   2.2646282    3.3367581
#      10.916321  ]
#    [ -8.56021      2.8471427   -1.3215172    0.8852517   -2.0804918
#      -3.5963378 ]
#    [  2.753674     2.2265363   -6.6377563    0.31403208  -4.410967
#     -12.985781  ]
#    [  3.334544     8.183901    -5.452185     2.1671524   -2.6073303
#       1.4117976 ]
#    [-10.743916    -0.12205492  -2.465022     1.8357716   -0.7580493
#       5.863118  ]]

#   [[  5.780536     5.502543    -6.188402    -8.727332     2.037358
#       8.804386  ]
#    [ -0.6780813   -0.8331849    3.3737197    0.50331825   3.2287257
#       3.5160232 ]
#    [ -0.37816492   7.750326     9.235642    -3.5693598    3.3006976
#       2.480992  ]
#    [ -2.7993839   -3.8555803    0.6955891    1.0687228    3.9249537
#      -8.181625  ]
#    [ -2.855411     7.0027833   -1.9879277    4.9676805    0.79862666
#      -2.574979  ]
#    [ -5.048195    -1.5901066    3.213563     5.7814965   -1.5875264
#       4.4493713 ]]

#   [[ 11.194473     2.3875396   -8.953257     5.496313     8.9386015
#      10.503973  ]
#    [  7.3260255    7.5123053    1.931275     1.292593     5.2304454
#      -1.9248575 ]
#    [ -5.956575    -0.7718687    3.1749086    3.4436338    0.7310545
#      -6.736043  ]
#    [ -8.251636     1.9667497    2.8968024    7.0446625   13.394309
#      -9.774336  ]
#    [ -3.1389017    1.6156917    4.1981893    0.21137719   6.7519054
#       0.5825275 ]
#    [  2.152685     3.785025    -0.3217218    9.873247     1.0275084
#       3.2635682 ]]]


#  [[[  0.34238312   1.33485     -7.3378572    6.389011    -2.6517735
#       0.8656527 ]
#    [ -4.8109646   -5.040539    -0.7027599   -2.8767078   -1.8255706
#      -4.72501   ]
#    [ -1.5542516   -4.87353      0.74099773   1.2531463   -0.514518
#      -5.3071194 ]
#    [  8.52045      3.9609578    3.3850121    3.0958884   -1.8593187
#       7.6077337 ]
#    [  2.1711981    3.5862842   -0.9093183    4.1398163   -1.7166017
#       3.0700877 ]
#    [ -0.52256006   9.289258    -3.0563173   -2.672453    -2.7715976
#      -0.1635197 ]]

#   [[ -1.7723857   -7.7136       5.3482704   -1.2680634   -4.252367
#      -2.3745992 ]
#    [ -3.2018895    0.78604233   2.2069125    2.4339545    2.9823952
#      -0.5857951 ]
#    [  6.256522     0.97448117   4.455102     7.45083      3.4377804
#      -0.03961059]
#    [  5.045198    -0.15618576  -1.3215283    9.753579   -14.767646
#       4.124437  ]
#    [  4.5069623   -1.9589518   -5.1757307    4.22674     -3.7731295
#       0.79273266]
#    [-11.193409     1.4097353    2.7773669    2.392736     1.3515433
#      -6.5734797 ]]

#   [[ -2.983257    -6.050316    -2.6358998   -0.1618195    0.86788005
#      -3.3714004 ]
#    [ 15.423567     3.3250935    7.598241     3.0266385    4.909357
#      -3.8688645 ]
#    [  5.156568     6.364912    -0.6268784    5.1238327   -3.2916286
#      -2.6995285 ]
#    [ -5.5202994   -2.4093535   -9.780079    -0.9002108    2.712024
#      -3.0801206 ]
#    [ -6.3237114   -6.5596023   -1.806522    -2.2107856   -3.0023136
#      -3.7933414 ]
#    [ -3.9296632   -4.5197005    5.0114183    2.636028    -4.446056
#       0.14866292]]

#   [[  1.5250858   -1.79179     -2.8472502    1.6935667   -2.9698017
#       2.21624   ]
#    [ 15.575943    11.604508    -2.3013222   -3.1636786    4.8202085
#      -2.6738641 ]
#    [ 12.817656     0.4115149   -7.722325    -1.1561264    1.5875452
#       3.2546213 ]
#    [ -9.323237    -5.6650295  -13.759613     5.308601     3.4300995
#      -1.0695885 ]
#    [ -3.2727609   -2.92773     -9.088984    -1.2734654    2.4471767
#      -3.8166525 ]
#    [-10.333296     1.5813684   -2.8698134    1.8504171    1.1989063
#      -5.3795    ]]]]
# Test passed! CPU and GPU results are close.

# 2D pattern test CPU:
#  [[[[8. 8. 8.]
#    [8. 8. 8.]
#    [8. 8. 8.]]]]
# 2D pattern test GPU:
#  [[[[8. 8. 8.]
#    [8. 8. 8.]
#    [8. 8. 8.]]]]
# Test passed! CPU and GPU results are close.

