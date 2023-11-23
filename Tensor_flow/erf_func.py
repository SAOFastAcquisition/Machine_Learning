import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from pathlib import Path


def flare(_x, _ampl, _pos, _width=10):
    return _ampl * np.exp(-0.5 * ((_x - _pos) / _width) ** 2)


def quiet_sun(_x=np.arange(-130, 130.1, 0.1), _level=1, _edge_l=-90, _edge_r=90, _width=10):
    _y1 = np.array([(1 + math.erf((a - _edge_l) / _width)) / 2 for a in _x])
    _y2 = - np.array([(1 + math.erf((a - _edge_r) / _width)) / 2 for a in _x])
    return (_y1 + _y2) * _level


def Gauss(_x, a, x0, s):
    return a * np.exp(-0.5 * ((_x - x0) / s) ** 2)


def nGauss(_x, *_p):
    n = (len(_p) - 5) // 3
    res = 0
    for i in range(n):
        res += Gauss(_x, _p[i * 3], _p[i * 3 + 1], _p[i * 3 + 2])
    res += (quiet_sun(_x, _p[-5], _p[-4], _p[-3], _p[-2]) + _p[-1])
    return res


def plotAllGauss(_x, _p):
    n = (len(_p) - 5) // 3
    for i in range(n):
        plt.plot(X, Gauss(X, _p[i * 3], _p[i * 3 + 1], _p[i * 3 + 2]), 'b', linewidth=0.5)
    plt.plot(X, quiet_sun(_x, _p[-5], _p[-4], _p[-3], _p[-2]) + _p[-1], 'b', linewidth=0.5)


def model():
    x = np.arange(-140, 140.1, 0.1)
    q_sun = quiet_sun(x)

    y3 = flare(x, 4, 20)
    y4 = flare(x, 2, -20) * 2
    y5 = flare(x, 3, -40)
    y6 = flare(x, 3, 60) / 2
    _y = q_sun + y3 + y4 + y5 + y6
    plt.plot(x, _y)
    plt.show()
    return x, _y


# начальные приближения для трёх гауссианов
ip0 = [5000., 160., 7.,
       4000., 180., 7.,
       7000., 190., 7.,

       14000., 200., 7.,
       4000.0, 225.8, 5.,
       1000., 245., 7.,
       3000., 256., 7.,
       8000., 266., 5.,
       2000., 280., 7.,
       5000., 150., 280., 10.0,
       500.]

# верхняя граница
top_limits = [6000, 170., 10.,
              7000., 190., 10.,
              9000., 200., 10.,
              20000, 215., 10.,
              7000., 235., 10.,
              1500., 255., 10.,
              5000., 260., 10.,
              10000., 270., 10.,
              4000., 285., 10,
              6000., 160., 300., 20.,
              1000.]

# X, Y = model()

path1 = Path("Tensor_flow/2023-02-17_04+16" + '_scan_freq.npy')
path2 = Path("Tensor_flow/2023-02-17_04+16" + '_time.npy')
Y = np.load(path1, allow_pickle=True)[6, 170:620]
X = np.load(path2, allow_pickle=True)[170:620]

plt.plot(X, Y, 'r+', markersize=3)

# интерполяция экспериментальных данных
p, cov = curve_fit(nGauss, X, Y, p0=ip0, bounds=(1, top_limits))
print("Параметры грауссианов: ")
for pl in p:
    print(pl)

    # погрешность вычислений
print("Станд. отклонение: ", np.std(Y - nGauss(X, *p)))
slope, ic, r_value, p_value, std_err = \
    stats.linregress(Y, nGauss(X, *p))

# вывод графиков
plt.ylabel('Антенная температура, К')
plt.xlabel('Время, сек')
plt.text(100, 15000, 'R$^2$=' + '%.4f' % r_value ** 2)
plotAllGauss(X, p)
plt.plot(X, nGauss(X, *p), 'g')
plt.grid('both')
plt.savefig('result.png')
plt.show()

pass
