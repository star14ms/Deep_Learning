# import multiprocessing as mp
# from multiprocessing import Pool

# print(mp.cpu_count())

# pool = Pool(4)

# print(dir(pool))

# a = ["a", "b", "b", "a"]
# print(a.index("a"))

###================================

# from common.layers import *

# x = np.random.randn(1, 1, 28, 28)
# W = np.random.randn(1, 1, 3, 3)
# b = np.random.randn(1)

# layer = Convolution(W, b, pad=1)

# y = layer.forward(x)
# print(y.shape)

# layer2 = Pooling(2, 2, stride=2, pad=0)

# z = layer2.forward(y)
# print(z.shape)

###================================

# A = {}
# print(type(A))

# import numpy as np

# B = np.zeros(3)
# print(B)

###================================

import time, os
from multiprocessing import Pool

def work_func(x):
    print("value %s is in PID : %s" % (x, os.getpid()))
    time.sleep(1)
    return x**5

def main():
    start = int(time.time())
    num_cores = 4
    pool = Pool(num_cores)
    print(pool.map(work_func, range(1,13)))
    print("***run time(sec) :", int(time.time()) - start)

if __name__ == "__main__":
    main()