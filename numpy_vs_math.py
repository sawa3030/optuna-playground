import math
import numpy as np
import time
import matplotlib.pyplot as plt

sizes = [1000, 10000, 100000, 1000000]
n_repeats = 10

numpy_times = []
math_times = []

for n in sizes:
    numpy_total = 0.0
    math_total = 0.0

    for _ in range(n_repeats):
        a = np.random.rand(n, 3)
        a = a[np.argsort(a[:, 0])]
        b = np.array([1, 1, 1])

        # use NumPy
        start = time.time()
        product = np.prod(b - a, axis=-1)
        end = time.time()
        numpy_total += (end - start)
        print("product with np.array:", product)

        # use Python loop
        start = time.time()
        product = 1.0
        for a_item, b_item in zip(a, b):
            product *= b_item - a_item
        end = time.time()
        math_total += (end - start)
        print("product with list:", product)

    numpy_times.append(numpy_total / n_repeats)
    math_times.append(math_total / n_repeats)

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
