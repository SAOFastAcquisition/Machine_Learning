import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from cic_filter import signal_filtering
import matplotlib
from matplotlib.ticker import MaxNLocator, ScalarFormatter, FixedLocator


def scan_normalize(_data):
    _temp = np.ma.masked_array(_data, np.isnan(_data))
    _aver = np.mean(_temp, axis=1).filled(np.nan)
    _dat = _data.T
    _dat /= _aver
    return _dat


def graph_contour_2d(*args):
    import matplotlib.font_manager as font_manager
    xval, yval, z, s, _info_txt, _current_file, _head = args
    x, y = np.meshgrid(xval, yval)
    # z = np.log10(z)
    a, b = z.min(), z.max()
    levels = MaxNLocator(nbins=15).tick_values(0.95, 1.10)
    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    cmap = plt.get_cmap('jet')

    fig, ax1 = plt.subplots(1, figsize=(12, 6))

    cf = ax1.contourf(x, y, z, levels=levels, cmap=cmap)
    # title1, title2, title3 = title_func(_current_file, _head)
    # fig.suptitle(title2 + ' ' + title1, y=1.0, fontsize=24)
    x_min = xval[1]
    y1 = yval[0] + (yval[-1] - yval[0]) * 0.05
    y2 = yval[0] + (yval[-1] - yval[0]) * 0.1
    fig.colorbar(cf, ax=ax1)
    # title1, title2 = pic_title()
    # ax1.set_title(title2 + ' ' + title1, fontsize=20)
    ax1.set_xlabel('Freq, MHz', fontsize=18)
    ax1.set_ylabel('Time, s', fontsize=18)

    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.5)
    plt.tick_params(axis='both', which='major', labelsize=16)

    plt.text(x_min, y1, _info_txt, fontsize=16)
    plt.text(x_min, y2, _info_txt, fontsize=16)

    # adjust spacing between subplots so `ax1` title and `ax0` tick labels
    # don't overlap
    fig.tight_layout()
    # add_path0 = path_to_pic(_current_file + '\\', 2, 'png')
    # fig.savefig(file_name0 + '\\' + add_path0)
    plt.show()
    # return fig, _current_file, 2, 'png'

    # Модуль проверки: формировалась ли ранее матрица спектра по времени и частоте
    # если - нет, то идем в extract(file_name0), если - да, то загружаем


def load_data(_path='2023-10-25_05-24_stocks.npy'):
    f_res = 3.904
    _data = np.load(Path('2023-02-16_14-24_stocks.npy'), allow_pickle=True)
    _y0 = _data[0]
    plt.plot(_y0[200])
    plt.show()

    _num_s = [_n for _n in range(4, 197)] + \
             [_n for _n in range(264, 279)] + \
             [_n for _n in range(312, 394)] + \
             [_n for _n in range(435, 501)]

    _yI = _data[0][550:1080, _num_s]
    _yV = _data[1][550:1080, _num_s]
    _y = (_yI + _yV) / 2
    # _y = _yI
    _time = np.array(_data[2][550:1080]) * 8.3886e-3
    _f = [1000 + f_res / 2 + f_res * _n for _n in _num_s]

    plt.plot(_time, _y[:, 10], label=f'freq = {np.ceil(_f[10])}')
    plt.grid('both')
    plt.title('2023-10-25_05-24 stocks I')
    plt.ylabel('Intensity (antenna temperature), K')
    plt.xlabel('Time, t')
    plt.legend(loc=2)
    plt.show()
    return _y, _time, _f


if __name__ == '__main__':
    data, time, freq = load_data()
    data_norm = scan_normalize(data)
    norm = np.mean(data_norm, axis=1)
    data_norm1 = data_norm.T / norm
    # plt.plot(np.array(freq) / 3.e4, data_norm1[10, :])
    n = [55, 75, 120]
    t0 = 70
    dt = 5
    plt.plot(np.array(freq) / 3.e4, norm, label=f'average')
    for i in range(10):
        a = np.mean(data_norm1[t0 + i * dt:t0 + (i + 1) * dt, :], axis=0)
        a = signal_filtering(a, 1.0) + 0.01 * i
        # plt.text(1500, 1.05-0.005 * i, f't = {np.ceil(time[t0 + dt / 2 + i * dt])} s')  + 0.01 * i
        plt.plot(np.array(freq) / 3.e4, a, label=f't = {np.ceil(time[t0 + dt + i * dt])} s')
        # plt.plot(np.array(freq), a, label=f't = {np.ceil(time[t0 + dt / 2 + i * dt])} s')

    plt.grid('both')
    plt.title('2023-10-25_05-24 stocks I')
    plt.ylabel('Normalized intensity')
    plt.xlabel('Wavenumber, cm_-1')
    # plt.xlabel('Frequency, MHz')
    plt.legend(loc=2)
    plt.show()

    s = 1
    _info_txt, _current_file, _head = 'a', 'b', 'c'
    graph_contour_2d(freq[125:181], time, data_norm1[:, 125:181], s, _info_txt, _current_file, _head)

