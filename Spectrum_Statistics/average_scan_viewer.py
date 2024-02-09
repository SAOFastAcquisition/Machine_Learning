import os
import gzip
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import pickle

from Support_scripts.plot_fig import path_to_pic, save_question, save_fig
from Support_scripts.paths_via_class import DataPaths
from Support_scripts.cic_filter import signal_filtering


class StrToCode:
    def __init__(self):
        self.file = open('codestr.py', 'w')

    def add_row(self, row):
        self.file.write(row + '\n')

    def run(self):
        self.file.close()
        os.system("python codestr.py")


def str_to_code(s):
    li = s.split('\n')
    S = StrToCode()
    for i in li:
        S.add_row(i)
    S.run()


def look_intensity_base(_angle, _arg, _n1, _n2, _base):
    _angle_selection = _base[(_base.angle < _angle + 10)
                             & (_base.angle > _angle - 10)
                             & (_base.azimuth.astype('int') < 34)]
    if not len(_angle_selection):
        print('         *** No request angle!!! ***')
        print(' Choose from below values')
        print(np.sort(np.array(_base.angle).transpose()))
        _angle = int(input('Sun position angle: '))
        _angle_selection = _base[(_base.angle < _angle + 10)
                                 & (_base.angle > _angle - 10)
                                 & (_base.azimuth.astype('int') < 34)]

    _x_l, _x_r, _param = _angle_selection.polar_left, \
                         _angle_selection.polar_right, \
                         _angle_selection.azimuth

    _arg0 = _arg[_n1:_n2]
    _xl = [s[_n1:_n2] for s in _x_l]
    _xr = [s[_n1:_n2] for s in _x_r]

    plot_intensities(_arg0, _xl, _xr, _param, _angle)


@save_fig
def plot_intensities(_arg, _x_l, _x_r, _param, _angle):
    _fig = plt.figure(figsize=[13, 8])
    #                           ****** Рисунок 1 левая поляризация ******
    _ax1 = plt.subplot(2, 1, 1)
    for _a, _b in zip(_x_l, _param):
        plt.plot(_arg, _a, label=f'left: azimuth = {_b} deg')

    _ax1.set_ylim(ymax=1.2)
    plt.grid('both')
    plt.title(str(path_treatment)[-16:] + ' Intensities L & R')
    _ax1.set_ylabel('Normalized Intensity L')
    plt.legend(loc=2)
    plt.text((_arg[0] + _arg[-1]) / 2, 1.16, f'Position angle on Sun {np.ceil(_angle)} arcs', size=12)
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.5)
    plt.minorticks_on()

    #                           ****** Рисунок 2 правая поляризация ******
    _ax2 = plt.subplot(2, 1, 2)
    for _a, _b in zip(_x_r, _param):
        plt.plot(_arg, _a, label=f'right: azimuth = {_b} deg')
    #                                       ************
    plt.grid('on')
    _ax2.set_ylabel('Normalized Intensity R')
    _ax2.set_ylim(ymax=1.2)
    plt.xlabel('Frequency, MHz')
    plt.legend(loc=2)
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.5)
    plt.minorticks_on()
    plt.show()

    return _fig, path_treatment, 7, 'png'


def plot_norm_intensities(_arg, _y_L, _y_R, _angles, _az, _polar):
    #                                ****** Рисунок ******
    _fig = plt.figure(figsize=[15, 9])
    gs = GridSpec(ncols=3, nrows=1, figure=_fig)
    r_LR = _y_L / _y_R
    #           ****** Рисунок 1 - спектры нормализованных интенсивностей LP & RP ******
    _ax1 = plt.subplot(gs[0, 0:2])

    _leg1 = [f'left: angle = {np.ceil(s)} arcs, azimuth = {u}' for s, u in zip(_angles, _az)]
    _leg2 = [f'right: angle = {np.ceil(s)} arcs, azimuth = {u}' for s, u in zip(_angles, _az)]
    _leg3 = [f'ratio L/R: angle = {np.ceil(s)} arcs azimuth = {u}' for s, u in zip(_angles, _az)]

    if _polar == 'left':
        _line1 = _ax1.plot(np.array(_arg), _y_L.T)
        _ax1.legend(_line1, _leg1, loc=2)
        plt.title(str(path_treatment)[-10:] + ' Intensities Left')
    elif _polar == 'right':
        _line2 = _ax1.plot(np.array(_arg), _y_R.T)
        _ax1.legend(_line2, _leg2, loc=2)
        plt.title(str(path_treatment)[-10:] + ' Intensities Right')
    elif _polar == 'both':
        _line1 = _ax1.plot(np.array(_arg), _y_L.T, '.-')
        _line2 = _ax1.plot(np.array(_arg), _y_R.T)
        # _ax1.legend(_line1 + _line2, _leg1 + _leg2, loc=2)
        _ax1.legend(_line1 + _line2, _leg1 + _leg2, bbox_to_anchor=(1, 1), loc="upper left")
        plt.title(str(path_treatment)[-10:] + ' Intensities L & R')
    else:
        _line1 = _ax1.plot(np.array(_arg), _y_L.T, '.-')
        _line3 = plt.plot(np.array(_arg), r_LR.T)
        # _ax1.legend(_line1 + _line3, _leg1 + _leg3, loc=2)
        _ax1.legend(_line1 + _line3, _leg1 + _leg3, bbox_to_anchor=(1, 1), loc="upper left")
    plt.grid('both')
    plt.ylabel('Normalized intensity')
    # ax.xlabel('Wave number, cm_-1') # [:, 5:edge1]
    plt.xlabel('Frequency, MHz')

    plt.grid(which='major', color='#666666', linestyle='-')
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.5)
    plt.minorticks_on()

    #                           ****** Рисунок 2 позиционный ******
    # _ax2 = plt.subplot(1, 3, 3)
    # plt.plot(angle, data_I, label=f'freq = {np.ceil(_arg[60])} MHz')
    # num_x = np.array(t0)
    # #               ****** Позиции на скане Солнца спектров интенсивностей ******
    # x1 = angle[num_x]
    # y1 = data_I[num_x]
    # plt.scatter(x1, y1)
    # #                           ****** Отображение ширины ДН ******
    # freq_lobe = 1.6
    # w_lobe = 224 / freq_lobe
    # for i in num_x:
    #     x = [angle[i] - w_lobe / 2, angle[i] + w_lobe / 2]
    #     y = [data_I[i], data_I[i]]
    #     plt.plot(x, y, 'g')
    # plt.text(-750, 8500, f'Ширина ДН {np.ceil(w_lobe)} arcs', )
    #
    # #                                       ************
    # plt.grid('on')
    # plt.title(str(path_treatment)[-16:] + ' Stokes I')
    # _ax2.yaxis.set_label_coords(-0.13, 3000)
    # _ax2.set_ylabel('Antenna Temperature, K')
    # plt.xlabel('Angle, arcs')
    # plt.legend(loc=2)
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
        _fig.savefig(path_pic)
        print('Picture is saved')
    return _fig, path_treatment, 7, 'png'


def load_base(_base_path):
    if os.path.exists(f'{str(_base_path)}.gz'):
        filename_out = f'{str(_base_path)}.gz'
        with gzip.open(filename_out, "rb") as fin:
            _norm_intensity_base = np.load(fin, allow_pickle=True)
    else:
        # _norm_intensity_base = np.load(_path1, allow_pickle=True)
        with open(path_stokes_base, 'rb') as inp:
            _norm_intensity_base = pickle.load(inp)

    return _norm_intensity_base


def base_filtering(_filter_str):
    _base = load_base()
    _filter_data = _base[str_to_code(_filter_str)]

    return _filter_data


def filter_azimuth(_azimuth, _base):
    _k = 0
    for _s in _azimuth:
        filt_cur = _base['azimuth'].astype('int') == _s
        if not _k:
            _filt_az = filt_cur
        else:
            _filt_az = filt_cur | _filt_az
        _k += 1

    return _filt_az


def filter_position(_angles):
    _i = 0
    for _s in _angles:
        _filt_cur = (base['angle'] < _s + 10.) & (base['angle'] > _s - 10.)
        if not _i:
            _filt_ang = _filt_cur
        else:
            _filt_ang = _filt_cur | _filt_ang
        _i += 1
    return _filt_ang


def freq_arg():
    # Для разрешения 3.906 МГц и количества отсчетов на 2 ГГц - 512:
    _num_s = [_n for _n in range(4, 197)] + \
             [_n for _n in range(264, 279)] + \
             [_n for _n in range(312, 394)] + \
             [_n for _n in range(435, 501)]

    _f = [1000 + f_res / 2 + f_res * _n for _n in _num_s]

    return np.array(_f), _num_s


if __name__ == '__main__':
    date_file = '2024-01-04'
    data_file = date_file + '_01+24'    # Используется этим скриптом для сохранения рисунков
    main_dir = date_file[0:4]
    data_dir = f'{date_file[0:4]}_{date_file[5:7]}_{date_file[8:10]}sun'
    path_obj = DataPaths(data_file, data_dir, main_dir)
    path_stokes = Path(str(path_obj.converted_data_file_path) + '_stocks.npy')
    path_stokes_base = Path(path_obj.converted_dir_path, 'norm_intensity_base.npy')
    # path_stokes_base = Path(path_obj.converted_dir_path, 'stokes_base.npy')
    path_treatment = path_obj.treatment_data_file_path
    polar = {2: 'right', 1: 'left', 3: 'both'}

    f_res = 7.8125 / 2
    edge0 = 193
    f1, f2 = 2250, 2520
    ind_polar = 3           # 1 - left, 2 - right, 3 - both polarization

    freq, num_s = freq_arg()
    s1 = np.where(freq > f1)[0][0]
    s2 = np.where(freq > f2)[0][0]
    base = load_base(path_stokes_base)

    # Посмотреть частоные зависимости нормализованных интенсивностей
    # при разных азимутах и фиксированном позиционном угле
    # look_intensity_base(-200, freq, s1, s2, base)

    # Посмотреть частотные зависимости нормализованных интенсивностей
    # для разных позиционных углов при фиксированном азимуте
    angles = [-450]
    az = [8, 4, 0]

    filt_ang = filter_position(angles)
    filt_az = filter_azimuth(az, base)
    filt = filt_az & filt_ang
    filt_base = base[filt]
    arg = freq[s1:s2]

    x_l, x_r, param_a, az0 = filt_base.polar_left, \
                             filt_base.polar_right, \
                             filt_base.angle, \
                             filt_base.azimuth

    xl = np.array([s[s1:s2] for s in x_l])
    xr = np.array([s[s1:s2] for s in x_r])
    param = [s for s in param_a]
    xl = signal_filtering(xl, 1)
    xr = signal_filtering(xr, 1)
    plot_norm_intensities(arg, xl, xr, param, az0, polar[ind_polar])
    pass
