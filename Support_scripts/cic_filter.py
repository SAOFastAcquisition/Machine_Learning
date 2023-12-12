import numpy as np               # Import numpy
import matplotlib.pyplot as plt  # Import matplotlib
# from scipy.signal import freqz
from scipy.signal import lfilter, filtfilt,  butter, freqs, remez, freqz
from scipy.fftpack import fft
# %matplotlib inline


class MafFilter:
    """
    Moving average filter:
    
    M - moving-average step (delay in comb stage)
    
    Parameters
    ----------
    x : np.array
        input 1-D signal
    """

    def __init__(self, x):
        self.x = x

    def maf_conv(self, m=2):
        """
        Calculate moving average filter via convolution

        Parameters
        ----------
        m : int
            moving average step
        """
        coe = np.ones(m) / m
        return np.convolve(self.x, coe, mode='same')

    def maf_fir(self, m=2):
        """
        Calculate moving average filter as FIR

        Parameters
        ----------
        m : int
            moving average step
        """
        return filtfilt(np.ones(m - 1), 1, self.x) / (m - 1) / (m - 1)

    def butter_iir(self, _delta_f, _order=5):
        """
        Calculate moving average filter as FIR

        Parameters
        ----------
        _order : filter order
        _delta_f : float
            cut off frequency
        """
        b, a = butter(_order, _delta_f)  # [low, high], btype='band'
        # Change to recursive form
        # a = [1, -1]
        # b = np.zeros(M)
        # b[0], b[-1] = a
        # w, h = freqs(b, a)
        # plt.semilogx(w, 20 * np.log10(abs(h)))
        # plt.show()
        return lfilter(b, a, self.x)

    def maf_iir(self, m=2):
        """
        Calculate moving average filter as FIR
        Parameters
        ----------
        m : int
            moving average step
        """
        # Change to recursive form
        a = [1, -1]
        b = np.zeros(m)
        b[0], b[-1] = a
        # w, h = freqs(b, a)
        # plt.semilogx(w, 20 * np.log10(abs(h)))
        # plt.show()
        return filtfilt(b, a, self.x)

    def remez_fir(self, _delta_f):
        bpass = remez(5, [0, 0.001, 0.002, 0.1, 0.5, 60], [0, 1, 0], Hz=122)
        freq, response = freqz(bpass)
        ampl = np.abs(response)

        # fig = plt.figure()
        # ax1 = fig.add_subplot(111)
        # ax1.semilogy(freq / (2 * np.pi), ampl, 'b-')  # freq in Hz
        # plt.show()

        return lfilter(bpass, 1, self.x)


def signal_filtering(_signal, delta_f):
    _p = 3
    _f = MafFilter(_signal)
    _filt = _f.maf_fir(_p)
    _signal1 = _signal.copy()
    _std = np.std(_signal - _filt)
    _f = MafFilter(_signal)
    _signal[abs(_signal[:] - _filt[:]) > 2 * _std] = _filt[abs(_signal[:] - _filt[:]) > 2 * _std]
    _std1 = np.std(_signal - _filt)
    _filt = _f.maf_fir(_p)
    _signal[abs(_signal[:] - _filt[:]) > 2 * _std] = _filt[abs(_signal[:] - _filt[:]) > 2 * _std]
    _std2 = np.std(_signal - _filt)
    _f = MafFilter(_signal)
    return _f.maf_fir(_p)


def model_signal():
    N = 300  # Number of samples
    # Input signal w/ noise:
    _sig = np.concatenate(
        (
            np.zeros(int(N / 2)),
            np.ones(int(N / 4)) * 7,
            np.zeros(int(N / 2)))
    )
    # Add some noise and peaks
    lns = _sig.size  # Size of signal
    np.random.seed(2)
    _sig += np.random.randn(lns)  # Add Gaussian noise
    rnd = np.random.randint(0, lns, 15)  # Add random numbers for index
    _sig[rnd] = 15  # Add peaks

    return _sig


if __name__ == '__main__':
    M = (2, 5, 500)  # Moving average step
    N = 300  # Number of samples
    LM = len(M)  # Size of M


    # sig1 = np.load('2021-12-26_03+12.npy', allow_pickle=True)
    # sig = sig1[0, :]
    sig = model_signal()
    lns = sig.size  # Size of signal

    # Calculate Moving Average filter:
    # filt = MafFilter(sig)
    res = np.zeros((lns, LM))
    for i in range(LM):
        # res[:, i] = filt.maf_conv(m=M[i])
        # res[:, i] = filt.maf_iir()  # m=M[i]
        res[:, i] = signal_filtering(sig, 0.0025)
    std = np.std(sig - res[:, 0])
    sig1 = np.zeros(lns)
    sig1 = sig.copy()
    sig[abs(sig[:] - res[:, 0]) > 2 * std] = res[abs(sig[:] - res[:, 0]) > 2 * std, 0]
    res1 = np.zeros(lns)
    res1 = signal_filtering(sig, 0.0025)
    std1 = np.std(res1 - sig)
    # Calculate Frequency responce:
    hfq = np.zeros((lns, LM))

    for j in range(LM):
        for i in range(lns):
            if i == 0:
                hfq[i, j] = 1
            else:
                hfq[i, j] = np.abs(np.sin(np.pi * M[j] * i / 2 / lns) / M[j] /
                                   np.sin(np.pi * i / 2 / lns))

    # Calculate spectrum of input signal:
    fft_sig = np.abs(fft(sig))
    fft_sig /= np.max(fft_sig)

    # Calculate spectrum of output signal:
    fft_out = np.zeros((lns, LM))
    for i in range(LM):
        fft_out[:, i] = np.abs(fft(res[:, i]))
        fft_out[:, i] /= np.max(fft_out[:, i])

    # Plot results:
    plt.figure(figsize=(12, 6), dpi=120)
    plt.subplot(3, 2, 1)
    plt.plot(sig, linewidth=1.25)
    plt.title('Input signal')
    plt.grid()
    # plt.xlim([0, 40000])

    plt.subplot(3, 2, 3)
    for i in range(LM):
        plt.plot(hfq[:, i], linewidth=1.25, label="M=%d" % M[i])
    plt.title('MA filter responce')
    plt.grid()
    plt.legend(loc=1)
    plt.xlim([0, lns - 1])

    plt.subplot(3, 2, 5)
    for i in range(LM-1):
        plt.plot(res[:, i], linewidth=1.0, label="M=%d" % M[i])
    plt.plot(res1, linewidth=1.0, label="M=%d" % M[LM-1])
    plt.title('Output signal')
    plt.grid()
    plt.legend(loc=2)
    plt.xlim([0, N - 1])

    for i in range(LM):
        plt.subplot(3, 2, 2 * i + 2)
        plt.plot(sig, '-', linewidth=0.5)
        plt.plot(sig1, '-', linewidth=0.5)
        plt.plot(res[:, i], linewidth=1.5)
        plt.title('Moving average, M = %d' % M[i])
        plt.grid()
        # plt.xlim([0, 40000])
        # plt.ylim([0, 1e20])

    plt.tight_layout()
    plt.show()




