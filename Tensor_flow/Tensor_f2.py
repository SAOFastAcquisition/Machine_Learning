import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def quiet_sun(_x, _level=1.0, _edge=90, _width=10):
    _y1 = (1 + tf.math.erf((_x + _edge) / _width)) / 2
    _y2 = -(1 + tf.math.erf((_x - _edge) / _width)) / 2
    return (_y1 + _y2) * _level


def flare(_x, _height, _pos, _width):
    _flare = _height * tf.exp(-0.5 * ((_x - _pos) / _width) ** 2)
    _flare_sum = tf.reduce_sum(_flare)
    return _flare_sum


x = tf.Variable(9.0, trainable=False)
edge = tf.Variable(20.0)
width = tf.Variable(3.0)
level = tf.Variable(1.0, trainable=True)
#
with tf.GradientTape(watch_accessed_variables=False, persistent=False) as tape:
    tape.watch(edge)
    y = quiet_sun(x, level, edge, width)

df = tape.gradient(y, [edge, width, level])
print(df[0], df[1], df[2], sep='\n')
del tape

height_n = np.array([1.0, 2.0, 3.0])
pos_n = np.array([-5.0, 0.0, 5.0])
width_n = np.array([1.0, 1.5, 1.0])
height = tf.Variable(height_n, dtype=float)
pos = tf.Variable(pos_n, dtype=float)
f_width = tf.Variable(width_n, dtype=float)

x_n = np.arange(-30, 30.1, 0.1)
flares_n = np.array([])
x_t = tf.Variable(-30, dtype=float)
while x_t < 30.1:
    flares = flare(x_t, height, pos, f_width) + quiet_sun(x_t, level, edge, width)
    flares_n = np.append(flares_n, flares.numpy())
    x_t.assign_add(0.1)
# print(flares_n)
plt.plot(x_n, flares_n)
plt.show()



