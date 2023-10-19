import numpy as np
import matplotlib.pyplot as plt
import random


def predict_poly(_x, _koeff):
    _res = 0
    _xx = [_x ** (len(_koeff) - n - 1) for n in range(len(_koeff))]

    for i, k in enumerate(_koeff):
        _res += k * _xx[i]

    return _res


def loss_func(_marg_i):
    return _marg_i ** 2
    # return (2 / (1 + np.exp(_marg_i)) - 1) ** 2
    # return 2 / (1 + np.exp(_marg_i))


def quality_func(_margin):
    return np.sum(_margin ** 2) / len(_margin)
    # return np.sum((2 / (1 + np.exp(_margin)) - 1) ** 2) / len(_margin)
    # return np.sum(2 / (1 + np.exp(_margin))) / len(_margin)


def grad_loss_func(_marg_i, _x_i):
    _grad = 2 * _marg_i * _x_i
    # _a = np.exp(_marg_i)
    # _grad = -4 * (2 / (1 + _a) - 1) * _a / (1 + _a) ** 2 * _x_i
    # _grad = -2 / (1 + _a) ** 2 * _a * _x_i
    return _grad


def rms_meth(_g, _gamma, _grad_loss):
    _g = _g * _gamma + (1 - _gamma) * _grad_loss * _grad_loss
    _nyu = etha * _grad_loss / (np.sqrt(_g) + eps0)
    return _nyu, _g


        # Model
x = np.arange(0, 10.1, 0.1)
y = 1 / (1 + 10 * np.square(x))

N = 8                   # Polynomial range
L = 0.8                 # Weight for polynomial coefficients
etha0 = 0.00000001       # Initial step changing of w
lambd = 0.0001          # Память при подсчете эмпирического риска
gamma = 0.9995          # Память при подсчете нормирующего множителя приращения коэффициентов
eps0 = 0.0001             # Слагаемое для избежания деления на "0"
n_iter = 150000

X = np.array([[a ** n for n in range(N + 1)] for a in x])   # Вектора-аргументы размеченной последовательности
# w = np.zeros(N + 1)                                         # Initial w
w = np.array([1 / 10 ** n for n in range(N + 1)])
Y = np.dot(X, w)                                            # Рассчитанные по w выходные значения

xx = x[:: 2]            # Аргументы размеченной последовательности
yy = y[:: 2]            # Значения (реакции) размеченной последовательности
X_train = np.array([[a ** n for n in range(N + 1)] for a in xx])    # Матрица признаков для размеченной послед
Y_predict = np.dot(X_train, w)                                      # Предсказанные начальные реакции
Q = np.sum(quality_func(Y_predict - yy))                            # Начальный эмпирический риск
q_plot = []
q = Q
g = q * q               # Нормировка для приращения "w"
len_train = len(yy)
for k in range(n_iter):
    i = random.randint(0, len_train - 1)  # Случайный номер в размеченной последовательности
    a = xx[i]
    x_i = np.array([a ** n for n in range(N + 1)])  # Его вектор признаков
    margin_i = np.dot(x_i, w) - yy[i]               # Отклонение предсказания от размеченного отклика
    Li = loss_func(margin_i)                        # Функция потерь для i-того члена размеч. последоват.
    etha = etha0 * (1 - k / n_iter)
    grad_Li = grad_loss_func(margin_i, x_i)  # Градиент функции потерь

    nyu, g = rms_meth(g, gamma, grad_Li)
    w = w - nyu                                     # etha0 * (1 - k / n_iter) * grad_Li

    # Q = Li * lambd + (1 - lambd) * Q
    margin_i = np.dot(x_i, w) - yy[i]
    eps = loss_func(margin_i)
    q = lambd * eps + (1 - lambd) * q           # Рекурентный подсчет функции потерь для всей выборки
    q_plot.append(q)
xx_c = x[1:: 2]
yy_c = y[1:: 2]

z_train = np.polyfit(xx, yy, N)
z_calc = w[::-1]
print(z_train)
# p = np.poly1d(z_train)
# y_fit = p(x)
y_fit = predict_poly(x, z_train)
y_calc = predict_poly(x, z_calc)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
ax0 = axes[0]
ax0.scatter(xx, yy, color='red')
ax0.scatter(xx_c, yy_c, color='blue')
ax0.plot(x, y_fit, color='green')
ax0.plot(x, y_calc, color='black')

# plt.xlim([0, 45])
# plt.ylim([0, 75])
# ax0.ylabel("длина")
# axo.xlabel("ширина")
ax0.grid(True)

axes[1].plot(q_plot, color='black')
axes[1].set_ylabel("функция ошибки")
axes[1].set_xlabel("иттерация")
axes[1].grid(True)

plt.show()
pass
