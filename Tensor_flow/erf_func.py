import math
import numpy as np
import pandas as pd
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
    return res + (quiet_sun(_x, _p[-5], _p[-4], _p[-3], _p[-2]) + _p[-1])


def plotAllGauss(_x, _p):
    n = (len(_p) - 5) // 3
    for i in range(n):
        plt.plot(X, Gauss(X, _p[i * 3], _p[i * 3 + 1], _p[i * 3 + 2]), 'b', linewidth=0.5)
    plt.plot(X, quiet_sun(_x, _p[-5], _p[-4], _p[-3], _p[-2]) + _p[-1], 'b', linewidth=0.5)


def model():
    """
    Возвращает модель Солнца = спокойное Солнце + 4 гауссианы. Второй аргумент во flare(x, 4, 20) -
    полуширина функции Гаусса, третий - ее сдвиг относительно "0"
    :return:
    """
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


def initial_cond():
    # начальные приближения для трёх гауссианов
    _ip0 = [5000., 160., 7.,
            4000., 180., 7.,
            7000., 190., 7.,
            14000., 200., 7.,
            4000.0, 225.8, 5.,
            50., 245., 7.,
            1000., 256., 7.,
            8000., 266., 5.,
            2000., 280., 7.,
            5000., 150., 285., 10.0,
            500.]

    # верхняя граница
    _top_limits = [20000, 170., 10.,
                   4000., 190., 18.,
                   17000., 200., 10.,
                   25000, 215., 10.,
                   9000., 235., 15.,
                   100., 255., 10.,
                   2000., 260., 10.,
                   15000., 270., 10.,
                   9000., 285., 10,
                   9000., 160., 300., 30.,
                   2000.]

    # нижняя граница
    _bottom_limits = [0, 140., 0.,
                      0., 170., 0.,
                      00., 180., 1.,
                      0, 200., 1.,
                      0., 220., 1.,
                      0., 245., 1.,
                      0., 250., 1.,
                      0., 260., 1.,
                      0., 275., 1,
                      0., 140., 285., 3.,
                      0.]
    return _ip0, _bottom_limits, _top_limits


def initial_cond1():
    # начальные приближения для трёх гауссианов
    _ip0 = [1000., 200., 7.,
            3000., 225., 7.,
            1000., 250., 7.,
            3000., 290., 7.,
            2000., 300., 5.,
            1000., 212., 7.,
            1000., 220., 7.,
            1000., 242., 7.,
            4000., 254., 5.,
            4000., 267., 7.,
            5000., 130., 280., 10.0,
            600.]

    # верхняя граница
    _top_limits = [20000., 150., 15.,
                   20000., 160., 18.,
                   20000., 170., 15.,
                   20000., 185., 15.,
                   20000., 195., 15.,
                   20000., 220., 15.,
                   20000., 225., 15.,
                   20000., 250., 15.,
                   20000., 260., 15.,
                   22000., 272., 15.,
                   22000., 140., 290., 25.,
                   6000.]

    # нижняя граница
    _bottom_limits = [0, 135., 1.,
                      0., 150., 1.,
                      0., 160., 1.,
                      0., 175., 1.,
                      0., 180., 1.,
                      0., 205., 1.,
                      0., 215., 1.,
                      0., 235., 1.,
                      0., 250., 1.,
                      0., 260., 1.,
                      0., 125., 275., 3.,
                      0.]
    return _ip0, _bottom_limits, _top_limits


def initial_cond0():
    # начальные приближения для трёх гауссианов
    _ip0 = [30000., 140., 7.,
            30000., 155., 7.,
            30000., 170., 7.,
            30000., 175., 7.,
            30000., 190., 7.,

            30000., 210., 7.,
            30000., 225., 7.,
            30000., 245., 7.,
            30000., 260., 7.,
            30000., 270., 7.,
            30000., 275., 7.,
            30000., 130., 280., 10.0,
            5000.]

    # верхняя граница
    _top_limits = [30000., 340., 15.,
                   30000., 340., 15.,
                   30000., 340., 15.,
                   30000., 340., 15.,
                   30000., 340., 15.,

                   30000., 340., 15.,
                   30000., 340., 15.,
                   30000., 340., 15.,
                   30000., 340., 15.,
                   30000., 340., 15.,
                   30000., 340., 15.,
                   30000., 185., 345., 30.,
                   6000.]

    # нижняя граница
    _bottom_limits = [0, 175., 1.,
                      0., 175., 1.,
                      0., 175., 1.,
                      0., 175., 1.,
                      0., 175., 1.,

                      0., 175., 1.,
                      0., 175., 1.,
                      0., 175., 1.,
                      0., 240., 1.,
                      0., 240., 1.,
                      0., 240., 1.,
                      0., 175., 335., 3.,
                      0.]
    return _ip0, _bottom_limits, _top_limits


def initial_cond2():
    # начальные приближения для трёх гауссианов
    _ip0 = [1000., 200., 7.,
            3000., 210., 7.,
            1000., 220., 7.,
            3000., 230., 7.,
            2000., 240., 5.,
            1000., 260., 7.,
            1000., 280., 7.,
            1000., 310., 7.,
            5000., 180., 340., 10.0,
            600.]

    # верхняя граница
    _top_limits = [20000., 220., 15.,
                   20000., 220., 18.,
                   20000., 240., 15.,
                   20000., 240., 15.,
                   20000., 340., 15.,
                   20000., 340., 15.,
                   20000., 340., 15.,
                   20000., 340., 15.,
                   22000., 190., 350., 25.,
                   6000.]

    # нижняя граница
    _bottom_limits = [0, 185., 1.,
                      0., 185., 1.,
                      0., 185., 1.,
                      0., 185., 1.,
                      0., 240., 1.,
                      0., 240., 1.,
                      0., 240., 1.,
                      0., 240., 1.,
                      0., 175., 335., 3.,
                      0.]
    return _ip0, _bottom_limits, _top_limits


def load_param():
    return np.load("2023-02-17_04+16" + '_decomp.npy', allow_pickle=True)


def load_data(_path):
    f_res = 3.904
    _data = np.load(_path, allow_pickle=True)
    _y0 = _data[0]
    _num_s = [98, 110, 120, 130, 140, 150, 160, 170, 180, 190]
    _y = _data[0][280:1450, _num_s]
    _time = np.array(_data[2][280:1450]) * 8.3886e-3
    _f = [1000 + f_res / 2 + f_res * _n for _n in _num_s]
    # plt.plot(_time, _y)
    # plt.show()
    return _y, _time, _f


def param_analise(_param=load_param()):
    # _param = load_param()
    _freq_mask = 1000 + np.array([98, 110, 120, 130, 140, 150, 160, 170, 180, 190]) * 3.904
    _arg = _freq_mask[-1::-1]
    _x0 = (_param[0:24:3, 3:] / _param[24, 3:]).transpose()
    plt.plot(_arg, _x0)
    plt.grid("both")
    plt.show()
    pass


if __name__ == '__main__':
    param = load_param()
    param_analise()
    # X, Y = model()  #Tensor_flow/
    path1 = Path('Tensor_flow/2022-06-18_01+28_stocks.npy')
    path2 = Path("Tensor_flow/2022-06-18_01+28" + '_time.npy')
    # path1 = Path("Tensor_flow/2023-02-17_04+16" + '_scan_freq.npy')

    # path2 = Path("Tensor_flow/2023-02-17_04+16" + '_time.npy')
    freq_mask = [1200, 1380, 1465, 1600, 1700, 2265, 2490, 2710, 2800, 2860]

    #               ***** Initial condition *****
    ip0, bottom_limits, top_limits = initial_cond2()
    load_data(path1)
    Yw, X1, f = load_data(path1)

    # X = np.load(path2, allow_pickle=True)[170:620]

    initial_data = {'bottom_limits': bottom_limits,
                    'top_limits': top_limits,
                    'initial_cond': ip0}
    lp = len(ip0)
    row_labels = []
    for i in range((lp - 5) // 3):
        row_labels += ['a' + str(i + 1), 'x0' + str(i + 1), 'w0' + str(i + 1)]
    row_labels += ['a0', 'xl', 'xr', 'ws', 'b0']

    #            ****** DataFrame for component parameters ******
    p_frame = pd.DataFrame(data=initial_data, index=row_labels)

    # num = np.shape(Yw)
    num = 10
    for i in range(num):
        # Параметры составляющих вычисляются от бОльших частот к меньшим
        lc = num - 1 - i
        Y1 = Yw[:, lc]
        num_a = np.isfinite(Y1)
        try:
            Y = Y1[num_a]
            X = X1[num_a]
            plt.plot(X, Y, 'r+', markersize=3)
            if np.size(Y) > np.size(Y1) // 2:
                #               ****** интерполяция экспериментальных данных ******
                p, cov = curve_fit(nGauss, X, Y, p0=ip0, bounds=(bottom_limits, top_limits))
                p_frame[str(np.ceil(f[lc]))] = p
                # ip0 = p
                print("Параметры грауссианов: ")
                for pl in p:
                    print(pl)

                #               ****** погрешность вычислений ******
                print("Станд. отклонение: ", np.std(Y - nGauss(X, *p)))
                slope, ic, r_value, p_value, std_err = \
                    stats.linregress(Y, nGauss(X, *p))

                #                   ****** вывод графиков ******
                plt.ylabel('Антенная температура, К')
                plt.xlabel('Время, сек')
                plt.text(140, 11000, 'R$^2$=' + '%.4f' % r_value ** 2)
                plt.text(140, 9000, f'f = {np.ceil(f[lc])} MHz')
                plotAllGauss(X, p)
                plt.plot(X, nGauss(X, *p), 'g')
                plt.grid('both')
                plt.savefig('result.png')
                plt.show()
            else:
                print('Too short sequence')
        except IndexError:
            pass
        except ValueError:
            np.save(Path("2023-02-17_04+16" + '_decomp'), p_frame)

    plt.plot(f[-1::-1], p_frame.loc['b0'][3:])
    plt.show()

    np.save(Path("2023-02-17_04+16" + '_decomp'), p_frame)
    pass
