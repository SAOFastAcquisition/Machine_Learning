import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from matplotlib.ticker import MaxNLocator, ScalarFormatter, FixedLocator
from tkinter import *
from tkinter import messagebox as mb


def save_fig(func):
    """ Функция-декоратор для сохранения в одноименную с файлом данных папку рисунков временных сканов и спектров
    в выделенные моменты времени."""

    def wrapper(*args):
        figure, file_name, flag, _format = func(*args)
        add_pass1 = path_to_pic(file_name, flag, _format)
        path = Path(file_name, add_pass1)
        if not os.path.exists(file_name):
            os.makedirs(file_name)
        figure.savefig(path, dpi=600)
        del figure
        flag_save = save_question()
        if flag_save == 'no':
            if os.path.isfile(path):
                os.remove(path)
                print('Picture is not saved')
            else:
                print('File not found')
        else:
            print('Picture is saved')
        return

    return wrapper


def save_question():
    root = Tk()
    answer = mb.askquestion(
        title="Save control",
        message="Save picture?")
    root.mainloop()
    del root
    return answer


@save_fig
def graph_contour_2d(*args):
    _x_val, _y_val, _z, _s, _info_txt, _pass_treat, _head = args
    x, y = np.meshgrid(_x_val, _y_val)

    _z = np.ma.masked_greater(_z, 1.5)
    _z = np.ma.masked_less(_z, 0.65)
    _z = np.log10(_z)
    _a, _b = np.min(np.ma.masked_array(_z, np.isnan(_z))), np.max(np.ma.masked_array(_z, np.isnan(_z)))
    levels = MaxNLocator(nbins=15).tick_values(_a, _b)
    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    cmap = plt.get_cmap('jet')

    fig, ax1 = plt.subplots(1, figsize=(12, 6))

    cf = ax1.contourf(x, y, _z, levels=levels, cmap='tab20b')
    # title1, title2, title3 = title_func(_pass_treat, _head)
    # fig.suptitle(title2 + ' ' + title1, y=1.0, fontsize=24)
    x_min = _x_val[1]
    y1 = _y_val[0] + (_y_val[-1] - _y_val[0]) * 0.05
    y2 = _y_val[0] + (_y_val[-1] - _y_val[0]) * 0.1
    fig.colorbar(cf, ax=ax1)
    # title1, title2 = pic_title()
    ax1.set_title('Normalized Antenna Temperature ' + str(_pass_treat)[-16:] + ' Stokes I', fontsize=20)
    ax1.set_xlabel('Freq, MHz', fontsize=18)
    ax1.set_ylabel('Angle, arcs', fontsize=18)

    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.5)
    plt.tick_params(axis='both', which='major', labelsize=16)

    plt.text(x_min, y1, _info_txt, fontsize=16)
    plt.text(x_min, y2, _info_txt, fontsize=16)

    # adjust spacing between subplots so `ax1` title and `ax0` tick labels
    # don't overlap
    fig.tight_layout()
    _add_path0 = path_to_pic(str(_pass_treat) + '\\', _s, 'png')

    plt.show()
    # fig.savefig(Path(_pass_treat, _add_path0))
    return fig, _pass_treat, _s, 'png'


@save_fig
def graph_3d(*args):
    from matplotlib import cm
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    xval, yval, _z, s, file_name, head = args

    _z = np.ma.masked_greater(_z, 1.5)
    _z = np.ma.masked_less(_z, 0.65)
    _z = np.log10(_z)
    x, y = np.meshgrid(xval, yval)
    ax.zaxis._set_scale('log')  # Расставляет tiks логарифмически
    # title1, title2, title3 = title_func(file_name, head)
    # ax.set_title(title2 + ' ' + title1, fontsize=20)
    # ax.text2D(0.05, 0.75, info_txt[0], transform=ax.transAxes, fontsize=16)
    # ax.text2D(0.05, 0.65, info_txt[1], transform=ax.transAxes, fontsize=16)
    ax.set_xlabel('Frequency, MHz', fontsize=16)
    ax.set_ylabel('Time, s', fontsize=16)
    ax.set_zlabel('Normalized Antenna Temperature, K', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=10)

    ax.plot_surface(x, y, _z, rstride=1, cstride=1, cmap='tab20b')  # cmap=cm.jet

    _format = 'png'
    plt.show()
    return fig, file_name, s, _format


def path_to_pic(file_path, flag, _format='png'):
    _comment = {'0': 'scan_00',
                '1': 'spectrum_00',
                '2': 'scan2D_00',
                '3': 'scan3D_00',
                '4': 'norm_stokes2D_00',
                '5': 'norm_stokes_00',
                '6': 'norm_stokes3D_00',
                '7': 'stokes_dyn_angle_00'}

    try:
        _add_pass0 = _comment[str(flag)]
    except:
        print('Check the "flag"')

    _l = len(_add_pass0)
    add_pass1 = _add_pass0 + '.' + _format
    path1 = Path(file_path, add_pass1)
    if not os.path.isfile(path1):
        pass
    else:
        while os.path.isfile(Path(file_path, add_pass1)):
            num = int(_add_pass0[_l - 2:_l]) + 1
            num_str = str(num)
            if num >= 10:
                _add_pass0 = _add_pass0[:_l - 2] + num_str
            else:
                _add_pass0 = _add_pass0[:_l - 2] + '0' + num_str
            add_pass1 = _add_pass0 + '.' + _format

    return add_pass1


if __name__ == '__main__':
    pass
