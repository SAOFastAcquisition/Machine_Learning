import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import pickle

from Support_scripts.cic_filter import signal_filtering
from Support_scripts.paths_via_class import DataPaths
from Support_scripts.plot_fig import graph_contour_2d, graph_3d, path_to_pic, save_question, save_fig
from Support_scripts.sun_az_spead import sun_az_speed

current_dir = Path.cwd()
sys.path.insert(0, current_dir)


def scan_self_normalize(_data):
    _aver_f = np.nanmean(_data, axis=1)
    _dat = _data.T / _aver_f
    _aver_t = np.nanmean(_dat, axis=1)
    _dat = _dat.T / _aver_t
    return _dat, _aver_f, _aver_t


def scan_intense_normalize(_data, _data_norm):
    _aver_f = np.nanmean(_data_norm, axis=1)
    _dat = _data.T / _aver_f
    _aver_t = np.nanmean(_dat, axis=1)
    _dat = _dat.T / _aver_t
    return _dat, _aver_f, _aver_t


def load_data(_n1, _n2, _path='2024-01-02_02+20_stocks.npy'):
    f_res = 7.8125 / 2  # 3.904
    _data = np.load(Path(_path), allow_pickle=True)
    _y0 = _data[0]
    # plt.plot(_y0[:, 150])
    # plt.grid('both')
    # plt.show()

    _num_s = [_n for _n in range(4, edge1)] + \
             [_n for _n in range(264, 279)] + \
             [_n for _n in range(312, 394)] + \
             [_n for _n in range(435, 501)]

    _yI = _data[0][_n1:_n2, _num_s]
    _yV = _data[1][_n1:_n2, _num_s]
    _yL = (_yI + _yV) / 2
    _yR = (_yI - _yV) / 2
    # _y = _yI
    _time = np.array(_data[2]) * 8.3886e-3
    _f = [1000 + f_res / 2 + f_res * _n for _n in _num_s]
    _n_scan = 150
    plt.plot(_time, _y0[:, _n_scan + 4])
    plt.plot(_time[_n1:_n2], _yI[:, _n_scan], label=f'freq = {np.ceil(_f[_n_scan])} MHz')
    plt.grid('both')

    plt.title(str(_path)[-27: -4])
    plt.ylabel('Intensity (antenna temperature), K')
    plt.xlabel('Time, t')
    plt.legend(loc=2)
    plt.show()
    return _yL, _yR, _time, _f


def time_to_angle(_time, _time_center, _path, _az):
    # _sun_width_time = 1058.  # Время прохождения солнечного диска через ДН
    # _sun_width = 14400  # Принятый угловой размер Солнца в арксек
    _scale = sun_az_speed(_path, _az)   # Угловая азимутальная скорость Солнца в арксек/сек
    _angle = [-(_t - _time_center) * _scale for _t in _time][-1::-1]

    return _angle


def sun_in_angle(_yl, _yr, _time, _time_c, _path, _az):
    _angle = np.array(time_to_angle(_time, _time_c, _path, _az))
    _yl = _yl[-1::-1, :]
    _yr = _yr[-1::-1, :]

    return _yl, _yr, _angle


def save_norm_intensity(_data_norm_L, _data_norm_R, _angle):
    # Если файла хранениия коэффициентов не существует, то создаем его, если существует - загружаем

    if not os.path.isfile(path_stokes_base):
        _intensity_dict = {'date': data_file[:10], 'azimuth': data_file[-3:], 'angle': int(_angle[0]),
                           'polar_left': [_data_norm_L[0]],
                           'polar_right': [_data_norm_R[0]]
                           }
        _norm_intensity_base = pd.DataFrame(_intensity_dict)
        with open(path_stokes_base, 'wb') as out:
            pickle.dump(_norm_intensity_base, out)
    else:
        with open(path_stokes_base, 'rb') as inp:
            _norm_intensity_base = pickle.load(inp)
    for i in range(41):

        idx = _norm_intensity_base.loc[(_norm_intensity_base.date == data_file[:10])
                                       & (_norm_intensity_base.azimuth == data_file[-3:])
                                       & (_norm_intensity_base.angle == int(_angle[i]))].index  #

        if not len(idx):
            _intensity_dict = {'date': data_file[:10], 'azimuth': data_file[-3:], 'angle': int(_angle[i]),
                               'polar_left': [_data_norm_L[i]],
                               'polar_right': [_data_norm_R[i]]
                               }
            _norm_intensity_base = pd.concat([_norm_intensity_base, pd.DataFrame(_intensity_dict)],
                                             axis=0, ignore_index=False)

        else:
            pass
    with open(path_stokes_base, 'wb') as out:
        pickle.dump(_norm_intensity_base, out)


@save_fig
def plot_norm_intensities(_arg, _y_L, _y_R):
    #                                ****** Рисунок ******
    _fig = plt.figure(figsize=[15, 9])
    gs = GridSpec(ncols=3, nrows=1, figure=_fig)

    #           ****** Рисунок 1 - спектры нормализованных интенсивностей LP & RP ******
    _ax1 = plt.subplot(gs[0, 0:2])

    _leg1 = [f'left: angle = {np.ceil(angle[s])} arcs' for s in num_angle]
    _leg2 = [f'right: angle = {np.ceil(angle[s])} arcs' for s in num_angle]

    _line1 = _ax1.semilogy(np.array(_arg), _y_L.T, '-.')
    _line2 = _ax1.semilogy(np.array(_arg), _y_R.T)
    _ax1.legend(_line1 + _line2, _leg1 + _leg2, loc=2)
    # _leg3 = [f'ratio L/R: angle = {np.ceil(angle[s])} arcs' for s in num_angle]
    # _line3 = plt.plot(np.array(freq), r_LR.T)
    # _ax1.legend(_line3, _leg3, loc=2)
    plt.grid('both')
    plt.title(str(path_treatment)[-16:] + ' Intensities L & R')
    plt.ylabel('Normalized intensity')
    # ax.xlabel('Wave number, cm_-1') # [:, 5:edge1]
    plt.xlabel('Frequency, MHz')

    plt.grid(which='major', color='#666666', linestyle='-')
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.5)
    plt.minorticks_on()

    #                           ****** Рисунок 2 позиционный ******
    _ax2 = plt.subplot(1, 3, 3)
    plt.plot(angle, data_I, label=f'freq = {np.ceil(_arg[60])} MHz')
    num_x = np.array(num_angle)
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
    _ax2.yaxis.set_label_coords(-0.13, 3000)
    _ax2.set_ylabel('Antenna Temperature, K')
    plt.xlabel('Angle, arcs')
    plt.legend(loc=2)
    plt.show()
    return _fig, path_treatment, 5, 'png'


if __name__ == '__main__':

    data_file = '2024-02-14_13-24'
    main_dir = data_file[0:4]
    data_dir = f'{data_file[0:4]}_{data_file[5:7]}_{data_file[8:10]}sun'

    path_obj = DataPaths(data_file, data_dir, main_dir)
    path_stokes = Path(str(path_obj.converted_data_file_path) + '_stocks.npy')
    path_stokes_base = Path(path_obj.converted_dir_path, 'norm_intensity_base.npy')
    path_treatment = path_obj.treatment_data_file_path

    save = 'y'  # Сохранять в базу нормированные интенсивности?

    n1, n2 = 439, 1116  # Начальный и конечный отсчеты времени диска Солнца
    t_center = 200  # Время кульминации от начала записи
    edge1 = 200  # Последний частотный отсчет перед первым режекторным фильтром
    edge0 = 180  # Последний частотный отсчет при визуализации данных
    start0 = 90  # Последний частотный отсчет при визуализации данных
    angle0 = [-1000. + 50 * i for i in range(41)]  # Начальное положение на диске Солнца центра ДН

    data_L, data_R, time, freq = load_data(n1, n2, path_stokes)
    # data_norm_L, aver_f1, aver_t1 = scan_self_normalize(data_L)
    # data_norm_R, aver_f2, aver_t2 = scan_self_normalize(data_R)
    data_norm_L, aver_f1, aver_t1 = scan_intense_normalize(data_L, data_R + data_L)
    data_norm_R, aver_f2, aver_t2 = scan_intense_normalize(data_R, data_R + data_L)
    l = len(time)
    data_norm_L, data_norm_R, angle_w = sun_in_angle(data_norm_L, data_norm_R, time, t_center,
                                                     Path(path_obj.primary_dir_path, '*.desc'),
                                                     int(data_file[-2::]))

    #              *** DATA filtering ***
    # data_norm_L = signal_filtering(data_norm_L, 1)
    # data_norm_R = signal_filtering(data_norm_R, 1)

    ratio_LR = data_norm_L / data_norm_R
    angle = angle_w[l - n2:l - n1]
    t0 = [np.where(angle >= s)[0][0] for s in angle0]  # Положение центра ДН в отсчетах угла

    if save == 'y':
        save_norm_intensity(data_norm_L[t0], data_norm_R[t0], angle[t0])

    data_I = data_L[-1::-1, 150] + data_R[-1::-1, 150]

    num_angle = t0
    a_L = data_norm_L[num_angle, start0:edge0]
    a_R = data_norm_R[num_angle, start0:edge0]
    r_LR = ratio_LR[num_angle, start0:edge0]

    # plot_norm_intensities(freq[start0:edge0], a_L, a_R)

    info_txt, head = 'Left polarization', 'a'
    # graph_contour_2d(freq[5:edge0], angle, data_norm_L[:, 5:edge0], 4, info_txt, path_treatment, head)
    info_txt, head = 'Right polarization', 'a'
    # graph_contour_2d(freq[5:edge0], angle, data_norm_R[:, 5:edge0], 4, info_txt, path_treatment, head)
    # graph_3d(freq[5:edge0], angle, data_norm_L[:, 5:edge0], 6, path_treatment, head)

    # Определение положений на Солнце для отображения на рис. нормированных интенсивностей
    theta = [-950, 950]
    num_angle = [np.where(angle >= s)[0][0] for s in theta]  # Положение центра ДН в отсчетах угла

    data_L_a, data_R_a, a = sun_in_angle(data_L, data_R, time, t_center,
                                         Path(path_obj.primary_dir_path, '*.desc'),
                                         int(data_file[-2::]))
    data_L_fig = data_L_a[num_angle, start0:edge0]
    data_R_fig = data_R_a[num_angle, start0:edge0]
    # plot_norm_intensities(freq[start0:edge0], data_L_fig, data_R_fig)
    a_L = data_norm_L[num_angle, start0:edge0]
    a_R = data_norm_R[num_angle, start0:edge0]
    plot_norm_intensities(freq[start0:edge0], a_L, a_R)