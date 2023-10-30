import math
import numpy as np
import matplotlib.pyplot as plt


def flare(_x, _pos, _ampl=1, _width=10):
    prob_density = _ampl * np.exp(-0.5 * ((_x - _pos) / _width) ** 2)
    return prob_density


def quiet_sun(_x=np.arange(-130, 130.1, 0.1), _level=1, _edge=90, _width=10):
    _y1 = np.array([(1 + math.erf((a + _edge) / _width)) / 2 for a in _x])
    _y2 = - np.array([(1 + math.erf((a - _edge) / _width)) / 2 for a in _x])
    return (_y1 + _y2) * _level


x = np.arange(-140, 140.1, 0.1)
q_sun = quiet_sun(x)

y3 = flare(x, 20, 4)
y4 = flare(x, -20, 2) * 2
y5 = flare(x, -40, 3)
y6 = flare(x, 60, 3) / 2
plt.plot(x, q_sun + y3 + y4 + y5 + y6)
plt.show()
