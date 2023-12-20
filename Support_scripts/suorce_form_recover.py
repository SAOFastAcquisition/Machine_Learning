import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft
from paths_via_class import DataPaths


def simplest_fig(_x, _y, _z):
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    axes[0].plot(_x, _y)
    axes[1].plot(_x, _z)
    axes[0].set_ylabel('Stocks_I')
    axes[1].set_ylabel('Stocks_V', color='darkred')
    axes[0].minorticks_on()
    axes[1].minorticks_on()
    axes[0].legend(loc='upper right')
    axes[0].grid()
    axes[1].grid()
    axes[0].grid(which='minor',
                 axis='_x',
                 color='k',
                 linestyle=':')
    axes[1].grid(which='minor',
                 axis='_x',
                 color='k',
                 linestyle=':')
    plt.show()


def gauss(_x, _s, _a=1, _x0=0):
    return _a*np.exp(-((_x-_x0)/_s)**2) / np.sqrt(2. * np.pi) / _s


def gauss_lobe(x, sigma):
    return np.exp(-((x / (.89 * sigma / (2 * np.sqrt(2 * np.log(2))))) ** 2) / 2) / (
            .89 * sigma / (2 * np.sqrt(2 * np.log(2))) * np.sqrt(2 * np.pi) * 300
    )


if __name__ == "__main__":
    current_primary_file = '2022-12-23_01+28'
    current_primary_dir = '2022_12_23sun'
    main_dir = '2022'
    main_dir = r'2021/Results'           # Каталог (за определенный период, здесь - за 2021 год)
    date = current_primary_dir[0:10]
    adr1 = DataPaths(current_primary_file, current_primary_dir, main_dir)
    converted_data_file_path = adr1.converted_data_file_path
    data_treatment_file_path = adr1.treatment_data_file_path
    data_saved_path = Path(data_treatment_file_path, 'Meshed_Spectrum', current_primary_file + '_meshed.npy')

    kt = 30
    delta_t0 = 8.3886e-3
    delta_t = delta_t0 * kt
    num1 = 12
    spectrum = np.load(data_saved_path)
    # spectrum = np.load('2022-06-18_01+28_stocks.npy', allow_pickle='True')
    spectrum_one = spectrum[0][:, num1]
    shape_spectrum = np.shape(spectrum[0])

    n_freq = shape_spectrum[1]
    delta_f = 2000/n_freq   # unit = 'MHz'
    f_lobe1 = 1000 + delta_f * num1
    f_lobe2 = 3000
    sigma_lobe1 = 252 / f_lobe1 * 1000
    sigma_lobe2 = 252 / f_lobe2 * 1000
    time = [i * delta_t for i in range(shape_spectrum[0])]
    # plt.plot(time, spectrum_one)
    # plt.show()

    center = 260                                    # sec time
    sun_wide = 160                                  # sec time
    center_arc = 0
    sun_wide_arc = 1920                                             # arcsec
    time_to_angle_coeff = sun_wide_arc / sun_wide                   # arcsec/sec
    angle_per_sample = time_to_angle_coeff * delta_t / 3600 / 57.2  # rad
    n_angle = int(6.28 / angle_per_sample)
    n_angle_center = int(n_angle // 2)
    angle_center = n_angle_center * 3600 * 57.2 * angle_per_sample
    delta_angle = delta_t * time_to_angle_coeff
    n_time_center = int(center / delta_t - 1)
    n_wide = int(150 / delta_t)
    # sun_centered = spectrum_one[n_time_center - n_wide - 1:n_time_center + n_wide]
    sun_centered = [0] * n_angle
    #           *** Совмещение точки кульминации с центральным отсчетом ***
    #                размещение скана Солнца посередине зоны Найквиста
    sun_centered[n_angle_center - n_wide - 1:n_angle_center + n_wide] = \
        spectrum_one[n_time_center - n_wide - 1:n_time_center + n_wide]
    #           *** Заполнение "хвостов" зоны Найквиста со сканом  ***
    for i in range(n_angle_center - n_wide):
        sun_centered[i] = sun_centered[n_angle_center - n_wide + 1]
        sun_centered[n_angle_center + n_wide + i] = sun_centered[n_angle_center + n_wide - 1]

    angle = np.array([t * angle_per_sample for t in range(n_angle)])
    main_lobe10 = gauss_lobe((angle - n_angle_center * angle_per_sample) * 3600 * 57.2, sigma_lobe1)
    main_lobe20 = gauss_lobe((angle - n_angle_center * angle_per_sample) * 3600 * 57.2, sigma_lobe2)
    # main_lobe10 = gauss(angle, 65 / 3600 / 57.2, 1, n_angle_center * angle_per_sample)
    # main_lobe20 = gauss(angle, 35 / 3600 / 57.2, 1, n_angle_center * angle_per_sample)

    num_arr = np.asarray(main_lobe10 > 1e-10).nonzero()
    r = [0.] * n_angle
    main_lobe2 = np.array([0.] * len(main_lobe20))
    main_lobe1 = np.array([0.] * len(main_lobe10))

    main_lobe1[num_arr] = main_lobe10[num_arr]
    main_lobe2[num_arr] = main_lobe20[num_arr]
    # plt.plot(angle[n_angle_center - 100:n_angle_center + 100] * 3600 * 57.2,
    #          main_lobe1[n_angle_center - 100:n_angle_center + 100])
    # plt.plot(angle[n_angle_center - 100:n_angle_center + 100] * 3600 * 57.2,
    #          main_lobe2[n_angle_center - 100:n_angle_center + 100])
    # plt.grid('on') # angle[n_angle - 2000:n_angle + 2000] * 3600 * 57.2,
    # plt.show()

    a_1 = fft(main_lobe1)   # Передаточная функция 1 телескопа (имеется)
    a_2 = fft(main_lobe2)  # Передаточная функция 2 телескопа (желательная)
    for i in range(len(main_lobe1)):
        if abs(a_1[i]) < 1e-4:
            a_1[i] = 1e-4
    for i in range(len(main_lobe2)):
        if abs(a_2[i]) < 1e-4:
            a_2[i] = 1e-4
    r = a_2 / a_1
    # plt.plot(abs(r))
    # plt.show()
    for i in range(n_angle):
        if abs(r[i]) > 10:
            r[i:-i] = 0
            r_inv = [0] * i
            r_inv = r[i:0:-1]
            r[n_angle - i - 1:-1] = r_inv
            break

    main_lobe_ift2 = ifft(a_2)
    spectrum_ft = fft(sun_centered)
    spectrum_ift = ifft(spectrum_ft * r)
    plt.plot(angle[n_angle_center - 800:n_angle_center + 800] * 3600 * 57.2 - angle_center,
             abs(spectrum_ift[n_angle_center - 800:n_angle_center + 800]), label=f'{f_lobe2} MHz')
    plt.plot(angle[n_angle_center - 800:n_angle_center + 800] * 3600 * 57.2 - angle_center,
             sun_centered[n_angle_center - 800:n_angle_center + 800], label=f'{int(f_lobe1)} MHz')
    plt.plot(angle[n_angle_center - 100:n_angle_center + 100] * 3600 * 57.2 - angle_center - 2000,
             main_lobe1[n_angle_center - 100:n_angle_center + 100] * 4e8, label=f'{int(f_lobe1)} MHz')
    plt.plot(angle[n_angle_center - 100:n_angle_center + 100] * 3600 * 57.2 - angle_center - 2000,
             main_lobe2[n_angle_center - 100:n_angle_center + 100] * 4e8, label=f'{int(f_lobe2)} MHz')
    plt.grid('on')
    plt.legend(loc="upper left")
    plt.show()
    # simplest_fig(angle, abs(spectrum_ift), sun_centered)
    pass
