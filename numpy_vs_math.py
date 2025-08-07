import math
import numpy as np
import time
import matplotlib.pyplot as plt

# 計測対象のサイズ
sizes = [1000, 10000, 100000, 1000000]
n_repeats = 10

numpy_times = []
math_times = []

for n in sizes:
    numpy_total = 0.0
    math_total = 0.0

    for _ in range(n_repeats):
        a = np.random.rand(n)
        b = ["max" if np.random.rand() > 0.5 else "min" for _ in range(n)]

        # NumPyベース
        start = time.time()
        a *= np.array([-1.0 if x == "max" else 1.0 for x in b])
        end = time.time()
        numpy_total += (end - start)

        a = np.random.rand(n)
        b = ["max" if np.random.rand() > 0.5 else "min" for _ in range(n)]

        # Pythonリストベース
        start = time.time()
        a *= [-1.0 if x == "max" else 1.0 for x in b]
        end = time.time()
        math_total += (end - start)

    numpy_times.append(numpy_total / n_repeats)
    math_times.append(math_total / n_repeats)

# プロット
plt.figure()
plt.plot(sizes, numpy_times, label="with np.array()", marker='o')
plt.plot(sizes, math_times, label="with list", marker='s')
plt.xlabel("Number of elements")
plt.ylabel("Average Time (s)")
plt.xscale("log")
plt.yscale("log")
plt.title("Performance Comparison")
plt.legend()
plt.grid(True)
plt.show()
