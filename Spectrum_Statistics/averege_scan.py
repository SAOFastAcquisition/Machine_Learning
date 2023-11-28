import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def scan_normalize(_data):
    _s = np.mean(_data)
    temp = np.ma.masked_array(_data, np.isnan(_data))
    _data /= _s
    return _data


def load_data(_path='2023-10-25_05-24_stocks.npy'):
    f_res = 3.904
    _data = np.load(Path('Spectrum_Statistics/2023-10-25_05-24_stocks.npy'), allow_pickle=True)
    _y0 = _data[0]
    _num_s = [_n for _n in range(4, 198)] + \
             [_n for _n in range(264, 279)] + \
             [_n for _n in range(312, 394)] + \
             [_n for _n in range(435, 501)]

    _y = _data[0][280:1330, _num_s]
    _time = np.array(_data[2][280:1330]) * 8.3886e-3
    _f = [1000 + f_res / 2 + f_res * _n for _n in _num_s]
    # plt.plot(_y[10])
    # plt.show()
    return _y, _time, _f


if __name__ == '__main__':
    data = load_data()
    data_norm = scan_normalize(data)
