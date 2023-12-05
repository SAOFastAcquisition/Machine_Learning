import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
from cic_filter import signal_filtering
from Tensor_flow.paths_via_class import DataPaths
import matplotlib
from matplotlib.ticker import MaxNLocator, ScalarFormatter, FixedLocator


current_dir = Path.cwd()
sys.path.insert(0, current_dir)


def scan_normalize(_data):
    _temp = np.ma.masked_array(_data, np.isnan(_data))
    _aver = np.mean(_temp, axis=1).filled(np.nan)
    _dat = _data.T
    _dat /= _aver
    return _dat


def graph_contour_2d(*args):
    import matplotlib.font_manager as font_manager
    xval, yval, z, _s, _info_txt, _pass_stocks, _head = args
    x, y = np.meshgrid(xval, yval)
    # z = np.log10(z)
    # z = np.ma.masked_greater(z, 1.5)
    # z = np.ma.masked_less(z, 0.85)
    _a, _b = np.min(np.ma.masked_array(z, np.isnan(z))), np.max(np.ma.masked_array(z, np.isnan(z)))
    levels = MaxNLocator(nbins=15).tick_values(_a, _b)
    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    cmap = plt.get_cmap('jet')

    fig, ax1 = plt.subplots(1, figsize=(12, 6))

    cf = ax1.contourf(x, y, z, levels=levels, cmap=cmap)
    # title1, title2, title3 = title_func(_pass_stoks, _head)
    # fig.suptitle(title2 + ' ' + title1, y=1.0, fontsize=24)
    x_min = xval[1]
    y1 = yval[0] + (yval[-1] - yval[0]) * 0.05
    y2 = yval[0] + (yval[-1] - yval[0]) * 0.1
    fig.colorbar(cf, ax=ax1)
    # title1, title2 = pic_title()
    ax1.set_title('Normalized Antenna Temperature ' + str(_pass_stocks)[-27: -4], fontsize=20)
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
    _add_path0 = path_to_pic(str(path_treatment) + '\\', _s, 'png')

    plt.show()
    fig.savefig(Path(path_treatment, _add_path0))
    # return fig, _pass_stoks, 2, 'png'


def path_to_pic(file_path, flag, _format='png'):
    if flag == 1:
        _add_pass0 = 'spectrum_00'
    elif flag == 2:
        _add_pass0 = 'colour2D_00'
    elif flag == 4:
        _add_pass0 = '2D_stokes_00'
    elif flag == 5:
        _add_pass0 = 'norm_stokes_00'
    elif flag == 3:
        _add_pass0 = 'pic3D_00'
    else:
        _add_pass0 = 'scan_00'

    l = len(_add_pass0)
    add_pass1 = _add_pass0 + '.' + _format
    path1 = Path(file_path, add_pass1)
    if not os.path.isfile(path1):
        pass
    else:
        while os.path.isfile(Path(file_path, add_pass1)):
            num = int(_add_pass0[l - 2:l]) + 1
            num_str = str(num)
            if num >= 10:
                _add_pass0 = _add_pass0[:l - 2] + num_str
            else:
                _add_pass0 = _add_pass0[:l - 2] + '0' + num_str
            add_pass1 = _add_pass0 + '.' + _format

    return add_pass1


def load_data(_path='2023-10-25_05-24_stocks.npy'):
    _n1, _n2 = 825, 1405
    f_res = 3.904
    _data = np.load(Path(_path), allow_pickle=True)
    _y0 = _data[0]
    # c = _y0[600, 270]
    plt.plot(_y0[:, 150])
    plt.show()

    _num_s = [_n for _n in range(4, 197)] + \
             [_n for _n in range(264, 279)] + \
             [_n for _n in range(312, 394)] + \
             [_n for _n in range(435, 501)]

    _yI = _data[0][_n1:_n2, _num_s]
    _yV = _data[1][_n1:_n2, _num_s]
    _y = (_yI + _yV) / 2
    # _y = _yI
    _time = np.array(_data[2][_n1:_n2]) * 8.3886e-3
    _f = [1000 + f_res / 2 + f_res * _n for _n in _num_s]

    plt.plot(_time, _y[:, 10], label=f'freq = {np.ceil(_f[10])}')
    plt.grid('both')

    plt.title(str(_path)[-27: -4])
    plt.ylabel('Intensity (antenna temperature), K')
    plt.xlabel('Time, t')
    plt.legend(loc=2)
    plt.show()
    return _y, _time, _f


if __name__ == '__main__':
    # path_stocks = '../Tensor_flow/2023-11-09_01+24_stocks.npy'
    data_file = '2022-06-18_07+04'
    main_dir = data_file[0:4]
    data_dir = f'{data_file[0:4]}_{data_file[5:7]}_{data_file[8:10]}sun'

    path_obj = DataPaths(data_file, data_dir, main_dir)
    path_stocks = Path(str(path_obj.converted_data_file_path) + '_stocks.npy')
    path_treatment = path_obj.treatment_data_file_path
    data, time, freq = load_data(path_stocks)
    data_norm = scan_normalize(data)
    norm = np.mean(data_norm, axis=1)
    data_norm1 = data_norm.T / norm
    # plt.plot(np.array(freq) / 3.e4, data_norm1[10, :])

    t0 = 40
    dt = 10
    fig, ax = plt.subplots(1, figsize=(14, 7))
    # ax.plot(np.array(freq[125:181]), norm[125:181], label=f'average')
    for i in range(15):
        a = np.nanmean(data_norm1[t0 + i * dt:t0 + (i + 1) * dt, 5:181], axis=0)
        a = signal_filtering(a, 1.0) + 0.015 * i
        # ax.text(1500, 1.05-0.005 * i, f't = {np.ceil(time[t0 + dt / 2 + i * dt])} s')  + 0.01 * i
        # ax.plot(np.array(freq[125:181]) / 3.e4, a, label=f't = {np.ceil(time[t0 + i * dt])} s') [125:181]
        ax.plot(np.array(freq)[5:181], a, label=f't = {np.ceil(time[t0 + i * dt])} s')

    ax.grid('both')
    ax.set_title(str(path_stocks)[-27: -4])
    ax.set_ylabel('Normalized intensity')
    # ax.xlabel('Wave number, cm_-1') [:, 125:181]
    ax.set_xlabel('Frequency, MHz')
    ax.legend(loc=2)
    add_path0 = path_to_pic(str(path_treatment) + '\\', 5, 'png')
    plt.show()
    fig.savefig(Path(path_treatment, add_path0))

    info_txt, head = 'a', 'b'
    graph_contour_2d(freq[125:181], time, data_norm1[:, 125:181], 4, info_txt, path_stocks, head)

