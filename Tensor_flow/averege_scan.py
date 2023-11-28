import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def scan_normalize(_data):
    temp = np.ma.masked_array(_data, np.isnan(_data))
    _aver = np.mean(temp, axis=1)
    aver1 = _aver.filled(np.nan)
    _dat = _data.T
    _dat /= aver1


    return _dat


def load_data(_path='2023-10-25_05-24_stocks.npy'):
    f_res = 3.904
    _data = np.load(Path('2023-10-25_05-24_stocks.npy'), allow_pickle=True)
    _y0 = _data[0]
    _num_s = [_n for _n in range(4, 198)] + \
             [_n for _n in range(264, 279)] + \
             [_n for _n in range(312, 394)] + \
             [_n for _n in range(435, 501)]

    _y = _data[0][550:1080, _num_s]
    _time = np.array(_data[2][550:1080]) * 8.3886e-3
    _f = [1000 + f_res / 2 + f_res * _n for _n in _num_s]
    # plt.plot(_y[:, 10])
    # plt.show()
    return _y, _time, _f



if __name__ == '__main__':
    data, time, freq = load_data()
    data_norm = scan_normalize(data)
    norm = np.mean(data_norm, axis=1)
    data_norm1 = data_norm.T / norm
    plt.plot(freq, data_norm1[5, :])
    n = [55, 75, 120]
    a = np.mean(data_norm1[175:185, :], axis=0)
    b = np.mean(data_norm1[170:180, :], axis=0)
    c = np.mean(data_norm1[165:175, :], axis=0)
    d = np.mean(data_norm1[160:170, :], axis=0)
    plt.plot(freq, a)
    plt.plot(freq, b)
    plt.plot(freq, c)
    plt.plot(freq, d)
    d1 = np.mean(data_norm1[280:290, :], axis=0)
    plt.plot(freq, d1)
    plt.text(2500, 2, f'f = {np.ceil(time[180])} s')
    plt.grid('both')
    plt.show()
