from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from Support_scripts.plot_fig import path_to_pic, save_question, save_fig
from Support_scripts.paths_via_class import DataPaths


@save_fig
def look_intensity_base(_angle, _arg):

    with open(path_stokes_base, 'rb') as inp:
        _norm_intensity_base = pickle.load(inp)     # Загрузка базы нормализованных интенсивностей за день

    _angle_selection = _norm_intensity_base[(_norm_intensity_base.angle < _angle + 10)
                                            & (_norm_intensity_base.angle > _angle - 10)]
    if not len(_angle_selection):
        print('         *** No request angle!!! ***')
        print(' Choose from below values')
        print(np.sort(np.array(_norm_intensity_base.angle).transpose()))
        _angle = int(input('Sun position angle: '))
        _angle_selection = _norm_intensity_base[(_norm_intensity_base.angle < _angle + 10)
                                                & (_norm_intensity_base.angle > _angle - 10)]

    _fig = plt.figure(figsize=[13, 8])
    #                           ****** Рисунок 1 левая поляризация ******
    _ax1 = plt.subplot(2, 1, 1)
    for _a, _b in zip(_angle_selection.polar_left, _angle_selection.azimuth):
        plt.plot(_arg, _a, label=f'left: azimuth = {_b} deg')

    _ax1.set_ylim(ymax=1.2)
    plt.grid('both')
    plt.title(str(path_treatment)[-16:] + ' Intensities L & R')
    _ax1.set_ylabel('Normalized intensity')
    plt.legend(loc=2)
    plt.text(1500, 1.18, f'Position angle on Sun {np.ceil(_angle)} arcs', size=12)
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.5)
    plt.minorticks_on()

    #                           ****** Рисунок 2 правая поляризация ******
    _ax2 = plt.subplot(2, 1, 2)
    for _a, _b in zip(_angle_selection.polar_right, _angle_selection.azimuth):
        plt.plot(_arg, _a, label=f'right: azimuth = {_b} deg')
    #                                       ************
    plt.grid('on')
    _ax2.set_ylabel('Normalized intensity')
    _ax2.set_ylim(ymax=1.2)
    plt.xlabel('Frequency, MHz')
    plt.legend(loc=2)
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.5)
    plt.minorticks_on()
    plt.show()

    return _fig, path_treatment, 7, 'png'


def freq_arg():

    # Для разрешения 3.906 МГц и количества отсчетов на 2 ГГц - 512:
    _num_s = [_n for _n in range(4, 197)] + \
             [_n for _n in range(264, 279)] + \
             [_n for _n in range(312, 394)] + \
             [_n for _n in range(435, 501)]

    _f = [1000 + f_res / 2 + f_res * _n for _n in _num_s]

    return np.array(_f)


if __name__ == '__main__':

    data_file = '2023-12-05_13-24'
    main_dir = data_file[0:4]
    data_dir = f'{data_file[0:4]}_{data_file[5:7]}_{data_file[8:10]}sun'
    path_obj = DataPaths(data_file, data_dir, main_dir)
    path_stokes = Path(str(path_obj.converted_data_file_path) + '_stocks.npy')
    path_stokes_base = Path(path_obj.converted_dir_path, 'norm_intensity_base.npy')
    path_treatment = path_obj.treatment_data_file_path

    f_res = 7.8125 / 2
    edge0 = 193
    freq = freq_arg()
    look_intensity_base(100, freq[90:edge0])
