import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
    # plt.plot(_y0[:, 150])
    # plt.grid('both')
    # plt.show()

    _num_s = [_n for _n in range(4, edge1)] #+ \
             # [_n for _n in range(264, 279)] + \
             # [_n for _n in range(312, 394)] + \
             # [_n for _n in range(435, 501)]

    _yI = _data[0][_n1:_n2, _num_s]
    _yV = _data[1][_n1:_n2, _num_s]
    _yL = (_yI + _yV) / 2
    _yR = (_yI - _yV) / 2
    # _y = _yI
    _time = np.array(_data[2]) * 8.3886e-3
    _f = [1000 + f_res / 2 + f_res * (_n + 4) for _n in _num_s]

    plt.plot(_time, _y0[:, 150 + 4])
    plt.plot(_time[_n1:_n2], _yI[:, 150], label=f'freq = {np.ceil(_f[150])} MHz')
    plt.grid('both')

    plt.title(str(_path)[-27: -4])
    plt.ylabel('Intensity (antenna temperature), K')
    plt.xlabel('Time, t')
    plt.legend(loc=2)
    plt.show()
    return _yL, _yR, _time, _f


def time_to_angle(_time, _time_center):
    _scale = 1960 / 162
    _angle = [-(_t - _time_center) * _scale for _t in _time][-1::-1]

    return _angle


def sun_in_angle(_yl, _yr, _time, _time_c):
    _angle = np.array(time_to_angle(_time, _time_c))
    _yl = _yl[-1::-1, :]
    _yr = _yr[-1::-1, :]

    return _yl, _yr, _angle


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
    angle0 = 750  # Начальное положение на диске Солнца центра ДН
    dt = 23  # Шаг в отсчетах угла для отображения нормализованных левой и правой поляризаций

    data_L, data_R, time, freq = load_data(n1, n2, path_stokes)
    data_norm_L, aver_f1, aver_t1 = scan_normalize(data_L)
    data_norm_R, aver_f2, aver_t2 = scan_normalize(data_R)
    l = len(time)
    data_norm_L, data_norm_R, angle_w = sun_in_angle(data_norm_L, data_norm_R, time, 264)
    angle = angle_w[l - n2:l - n1]
    t0 = np.where(angle >= angle0)[0][0]    # Начальное положение центра ДН в отсчетах угла

    data_I = data_L[-1::-1, 150] + data_R[-1::-1, 150]

    #                                ****** Рисунок ******
    fig = plt.figure(figsize=[15, 9])
    gs = GridSpec(ncols=3, nrows=1, figure=fig)

    #           ****** Рисунок 1 - спектры нормализованных интенсивностей LP & RP ******
    ax1 = plt.subplot(gs[0, 0:2])
    # ax.plot(np.array(freq[5:edge0]), norm[5:edge0], label=f'average')
    for i in range(3):
        a_L = data_norm_L[t0 + i * dt, 90:edge0]
        a_R = data_norm_R[t0 + i * dt, 90:edge0]
        a_L = signal_filtering(a_L, 1.0)
        a_R = signal_filtering(a_R, 1.0)

        # ax.plot(np.array(freq[5:edge0]) / 3.e4, a, label=f't = {np.ceil(time[t0 + i * dt])} s') [5:edge0]
        plt.plot(np.array(freq)[90:edge0], a_L, '-.',  label=f'left: angle = {np.ceil(angle[t0 + i * dt])} arcs')
        plt.plot(np.array(freq)[90:edge0], a_R, label=f'right: angle = {np.ceil(angle[t0 + i * dt])} arcs')

    plt.grid('both')
    plt.title(str(path_treatment)[-16:] + ' Intensities L & R')
    plt.ylabel('Normalized intensity')
    # ax.xlabel('Wave number, cm_-1') # [:, 5:edge1]
    plt.xlabel('Frequency, MHz')
    plt.legend(loc=2)
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.5)
    plt.minorticks_on()

    #                           ****** Рисунок 2 позиционный ******
    ax2 = plt.subplot(1, 3, 3)
    plt.plot(angle, data_I,  label=f'freq = {np.ceil(freq[150])} MHz')
    num_x = np.array([t0 + i * dt for i in range(3)])
    #               ****** Позиции на скане Солнца спектров интенсивностей ******
    x1 = angle[num_x]
    y1 = data_I[num_x]
    plt.scatter(x1, y1)
    #                           ****** Отображение ширины ДН ******
    freq_lobe = 1.6
    w_lobe = 224 / freq_lobe
    for i in num_x:
        x = [angle[i] - w_lobe / 2, angle[i] + w_lobe / 2]
        y = [data_I[i], data_I[i]]
        plt.plot(x, y, 'g')
    plt.text(-750, 8500, f'Ширина ДН {np.ceil(w_lobe)} arcs', )

    #                                       ************
    plt.grid('on')
    plt.title(str(path_treatment)[-16:] + ' Stokes I')
    ax2.yaxis.set_label_coords(-0.13, 3000)
    ax2.set_ylabel('Antenna Temperature, K')
    plt.xlabel('Angle, arcs')
    plt.legend(loc=2)
    plt.show()

    #                       ****** Формирование адреса и сохранение рисункa ******
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

    info_txt, head = 'Right polarization', 'a'
    # graph_contour_2d(freq[5:edge0], angle, data_norm_R[:, 5:edge0], 4, info_txt, path_treatment, head)
    # graph_3d(freq[5:edge0], time, data_norm[:, 5:edge0], 6, path_treatment, head)
