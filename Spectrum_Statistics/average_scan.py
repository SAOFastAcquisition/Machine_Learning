import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from Support_scripts.cic_filter import signal_filtering
from Support_scripts.paths_via_class import DataPaths
from Support_scripts.plot_fig import graph_contour_2d, graph_3d, path_to_pic, save_question

current_dir = Path.cwd()
sys.path.insert(0, current_dir)


def scan_normalize(_data):
    _aver_f = np.nanmean(_data, axis=1)
    _dat = _data.T / _aver_f
    _aver_t = np.nanmean(_dat, axis=1)
    _dat = _dat.T / _aver_t
    return _dat, _aver_f, _aver_t


def load_data(_n1, _n2, _path='2023-10-25_05-24_stocks.npy'):
    f_res = 3.904
    _data = np.load(Path(_path), allow_pickle=True)
    _y0 = _data[0]
    # c = _y0[600, 270]
    plt.plot(_y0[:, 150])
    plt.grid('both')
    plt.show()

    _num_s = [_n for _n in range(4, edge1)] #+ \
             # [_n for _n in range(264, 279)] + \
             # [_n for _n in range(312, 394)] + \
             # [_n for _n in range(435, 501)]

    _yI = _data[0][_n1:_n2, _num_s]
    _yV = _data[1][_n1:_n2, _num_s]
    _yL = (_yI + _yV) / 2
    _yR = (_yI - _yV) / 2
    # _y = _yI
    _time = np.array(_data[2][_n1:_n2]) * 8.3886e-3
    _f = [1000 + f_res / 2 + f_res * _n for _n in _num_s]

    plt.plot(_time, _yR[:, 150], label=f'right freq = {np.ceil(_f[150])}')
    plt.grid('both')

    plt.title(str(_path)[-27: -4])
    plt.ylabel('Intensity (antenna temperature), K')
    plt.xlabel('Time, t')
    plt.legend(loc=2)
    plt.show()
    return _yL, _yR, _time, _f


if __name__ == '__main__':
    # path_stokes = '../Tensor_flow/2023-11-09_01+24_stocks.npy'
    data_file = '2022-06-18_01+28'
    main_dir = data_file[0:4]
    data_dir = f'{data_file[0:4]}_{data_file[5:7]}_{data_file[8:10]}sun'

    path_obj = DataPaths(data_file, data_dir, main_dir)
    path_stokes = Path(str(path_obj.converted_data_file_path) + '_stocks.npy')
    path_treatment = path_obj.treatment_data_file_path

    n1, n2 = 750, 1330  # Начальный и конечный отсчеты времени диска Солнца
    edge1 = 197         # Последний частотный отсчет перед первым режекторным фильтром
    edge0 = 197         # Последний частотный отсчет при визуализации данных

    data_L, data_R, time, freq = load_data(n1, n2, path_stokes)
    data_norm_L, aver_f1, aver_t1 = scan_normalize(data_L)
    data_norm_R, aver_f2, aver_t2 = scan_normalize(data_R)
    # plt.plot(np.array(freq) / 3.e4, data_norm[10, :])
    # data_V = data_V.T / aver_f
    # data_V = data_V.T / aver_t
    # data_norm = data_V
    t0 = 50
    dt = 5
    fig, ax = plt.subplots(1, figsize=(14, 7))
    # ax.plot(np.array(freq[5:edge0]), norm[5:edge0], label=f'average')
    for i in range(3):
        a_L = np.nanmean(data_norm_L[t0 + i * dt:t0 + (i + 1) * dt, 90:edge0], axis=0)
        a_L = np.nanmean(data_norm_L[t0 + i * dt:t0 + (i + 1) * dt, 90:edge0], axis=0)
        a_L = signal_filtering(a_L, 1.0)  # + 0.015 * i
        a_R = np.nanmean(data_norm_R[t0 + i * dt:t0 + (i + 1) * dt, 90:edge0], axis=0)
        a_R = np.nanmean(data_norm_R[t0 + i * dt:t0 + (i + 1) * dt, 90:edge0], axis=0)
        a_R = signal_filtering(a_R, 1.0)  # + 0.015 * i

        # ax.plot(np.array(freq[5:edge0]) / 3.e4, a, label=f't = {np.ceil(time[t0 + i * dt])} s') [5:edge0]
        ax.plot(np.array(freq)[90:edge0], a_L, label=f'left: t = {np.ceil(time[t0 + i * dt])} s')
        ax.plot(np.array(freq)[90:edge0], a_R, label=f'right: t = {np.ceil(time[t0 + i * dt])} s')

    ax.grid('both')
    ax.set_title(str(path_treatment)[-16:] + ' Stokes I')
    ax.set_ylabel('Normalized intensity')
    # ax.xlabel('Wave number, cm_-1') # [:, 5:edge1]
    ax.set_xlabel('Frequency, MHz')
    ax.legend(loc=2)
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.5)
    plt.minorticks_on()
    plt.show()

    add_path0 = path_to_pic(path_treatment, 5, 'png')
    path_pic = Path(path_treatment, add_path0)
    flag_save = save_question()
    if flag_save == 'no':
        if os.path.isfile(path_pic):
            os.remove(path_pic)
            print('Picture is not saved')
        else:
            print('File not found')
    else:
        fig.savefig(path_pic)
        print('Picture is saved')

    info_txt, head = 'a', 'b'
    graph_contour_2d(freq[5:edge0], time, data_norm_R[:, 5:edge0], 4, info_txt, path_treatment, head)
    # graph_3d(freq[5:edge0], time, data_norm[:, 5:edge0], 6, path_treatment, head)
