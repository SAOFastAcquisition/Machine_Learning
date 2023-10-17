import numpy as np
import matplotlib.pyplot as plt
import random


def loss_func(_marg_i):
    return 2 / (1 + np.exp(_marg_i))


def quality_func(_margin):
    return np.sum(2 / (1 + np.exp(_margin))) / len(_margin)


def grad_loss_func(_marg_i, _x_i, _y_i):
    _a = np.exp(_marg_i)
    _grad = -2 / (1 + _a)**2 * _a * _x_i * _y_i
    return _grad


def rms_meth(_g, _gamma, _grad_loss):

    _g = _g * _gamma + (1 - _gamma) * _grad_loss * _grad_loss
    _nyu = etha0 * _grad_loss / (np.sqrt(_g) + eps)
    return _nyu


etha0 = 0.02
eps = 0.1
n = 2000
n_mem = 300
gamma = 0.01     # (n_mem - 1) / (n_mem + 1)
lambd = 0.01    # 2 / (n + 1)
x_train = [[10, 50], [20, 30], [25, 30], [20, 60], [15, 70], [40, 40], [30, 45], [20, 45], [40, 30], [7, 35]]
x_train = np.array([s + [1] for s in x_train])
y_train = np.array([-1, 1, 1, -1, -1, 1, 1, -1, 1, -1])
len_train = len(y_train)

w = np.array([0 for i in range(3)])                                 # Инициализация начальных коэффициентов
margin_init = [np.dot(x, w) * y for x, y in zip(x_train, y_train)]  # Строка отступов для размеченной выборки
q = quality_func(margin_init)                                      # Значение функции потерь при заданных w
g = q * q
q_plot = []
nyu = 0
etha = etha0
for j in range(n):
    i = random.randint(0, 9)
    margin = np.dot(x_train[i], w) * y_train[i]                 # Отступ для i-того члена выборки
    grad_loss = grad_loss_func(margin, x_train[i], y_train[i])  # Функция потерь для него же по SGD
    etha = etha0 * (1 - j / n)        # eta0 * np.exp(- j / n)     # Модификация шага по ходу работы
    nyu = rms_meth(g, gamma, grad_loss)                  # Изменение w с применением метода momentum
    w = w - nyu
    margin = np.dot(x_train[i], w) * y_train[i]
    eps = loss_func(margin)
    q = lambd * eps + (1 - lambd) * q           # Рекурентный подсчет функции потерь для всей выборки
    q_plot.append(q)

    pass

line_x = list(range(max(x_train[:, 0])))  # формирование графика разделяющей линии
line_y = [- w[0] / w[1] * x - w[2] / w[1] for x in line_x]

x_0 = x_train[y_train == 1]  # формирование точек для 1-го
x_1 = x_train[y_train == -1]  # и 2-го классов

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
ax = axes[0]
# ax0 = fig.add_subplot(1, 2, 2)

ax.scatter(x_0[:, 0], x_0[:, 1], color='red')
ax.scatter(x_1[:, 0], x_1[:, 1], color='blue')
ax.plot(line_x, line_y, color='green')

ax.set_xlim(0, 45)
ax.set_ylim(0, 75)
ax.set_ylabel("длина")
ax.set_xlabel("ширина")
ax.grid(True)

axes[1].plot(q_plot, color='black')
axes[1].set_ylabel("иттерация")
axes[1].set_xlabel("функция ошибки")
axes[1].grid(True)

plt.show()
