import numpy as np
import matplotlib.pyplot as plt
import random

x_train = [[10, 50], [20, 30], [25, 30], [20, 60], [15, 70], [40, 40], [30, 45], [20, 45], [40, 30], [7, 35]]
x_train = np.array([s + [1] for s in x_train])
y_train = np.array([-1, 1, 1, -1, -1, 1, 1, -1, 1, -1])
len_train = len(y_train)

w = np.array([random.random() for i in range(3)])
m = [np.dot(x, w) for x in x_train]
q_init = sum(m) / len_train

pass