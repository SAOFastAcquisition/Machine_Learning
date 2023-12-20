import numpy as np
import matplotlib.pyplot as plt


def gauss_lobe(_x, _sigma):
    """
    Принимает значение угла в арксекундах и параметр ширины главного лепестка в акксекундах,
    возвращает апроксимацию главного лепестка функцией Гаусса
    :param _x: значение угла в арксекундах
    :param _sigma: параметр ширины главного лепестка
    :return: нормированная по мощности к единице ДН
    """
    return np.exp(-((_x / (.89 * _sigma / (2 * np.sqrt(2 * np.log(2))))) ** 2) / 2) / (
            .89 * _sigma / (2 * np.sqrt(2 * np.log(2))) * np.sqrt(2 * np.pi) * 300
    )


def sigma(_f):
    """
    Возвращает параметр для расчета главного лепестка ДН на частоте _f
    :param _f: частота в ГГц
    :return: параметр главного лепестка в арксекундах
    """
    return 252. / _f


if __name__ == '__main__':

    #           Расчет параметра ширины главного лепестка
    sigma_s = np.load('sigma.npy', allow_pickle='True')
    freq_s = np.load('frequencies.npy', allow_pickle='True')
    freq = np.arange(1., 3., 0.01)
    sigma_f = sigma(freq)
    #                       ****************
    #           Расчет главного лепестка ДН (х - угол в арксекундах) на частоте f (ГГЦ)
    x = np.arange(-1800., 1800., 10.0)
    f = 1.
    main_lobe = gauss_lobe(x, sigma(f))
    #                       ****************

    plt.plot(x, main_lobe)
    plt.grid('on')
    plt.show()

    plt.plot(freq, sigma_f)
    plt.grid('on')
    plt.show()

