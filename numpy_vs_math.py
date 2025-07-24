import math
import numpy as np
import time

for n in [1000, 10000, 100000, 1000000]:
    a = np.random.rand(n)
    b = ["max" if np.random.rand() > 0.5 else "min" for _ in range(n)]

    start = time.time()
    a *= np.array([-1.0 if x == "max" else 1.0 for x in b])
    end = time.time()
    print("numpy:", n, end - start)


    a = np.random.rand(n)
    b = ["max" if np.random.rand() > 0.5 else "min" for _ in range(n)]

    start = time.time()
    a *= [-1.0 if x == "max" else 1.0 for x in b]
    end = time.time()
    print("math:", n, end - start)