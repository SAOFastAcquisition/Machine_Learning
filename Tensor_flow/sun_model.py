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
    # _flare_sum = tf.reduce_sum(_flare)
    return _flare


def sun_with_noise(_x, _level=1.0, _edge=90, _width=10):
    _len = int(tf.size(_x).numpy())
    _sun = quiet_sun(_x, _level, _edge, _width)
    _noise = tf.random.normal(shape=[_len], stddev=0.01)
    return _sun + _noise


end = 30.0
step_x = 0.1
x_n = np.arange(-end, end + 0.1, step_x)
data = sun_with_noise(tf.Variable(x_n, dtype=float), 1.0, 20.0, 2.0) + \
       flare(tf.Variable(x_n, dtype=float), 1.5, 3.0, 1.0)

#                   *** Model ***
level_mod = tf.Variable(1.5)
edge_mod = tf.Variable(15.0)
width_mod = tf.Variable(2.5)
height_mod = tf.Variable(2.0)
pos_mod = tf.Variable(5.0)
width1_mod = tf.Variable(0.5)
model = quiet_sun(x_n, level_mod, edge_mod, width_mod) + flare(x_n, height_mod, pos_mod, width1_mod)

EPOCHS = 800
learning_rate = 1.5
learning_rate1 = 0.015
loss_dyn = np.array([])
for n in range(EPOCHS):
    with tf.GradientTape() as t:
        f = quiet_sun(x_n, level_mod, edge_mod, width_mod) + flare(x_n, height_mod, pos_mod, width1_mod)
        loss = tf.reduce_mean(tf.square(data - f))
        loss_dyn = np.append(loss_dyn, loss.numpy())
    dp1, dp2, dp3, dp4, dp5, dp6 = t.gradient(loss, [level_mod, edge_mod, width_mod, height_mod, pos_mod, width1_mod])

    level_mod.assign_sub(learning_rate * dp1)
    edge_mod.assign_sub(learning_rate * dp2)
    width_mod.assign_sub(learning_rate * dp3)
    height_mod.assign_sub(learning_rate1 * dp4)
    pos_mod.assign_sub(learning_rate1 * dp5)
    width1_mod.assign_sub(learning_rate1 * dp6)

sun_calc = quiet_sun(x_n, level_mod, edge_mod, width_mod) + flare(x_n, height_mod, pos_mod, width1_mod)
print(level_mod, edge_mod, width_mod, height_mod, pos_mod, width1_mod, sep='\n')
plt.plot(x_n, sun_calc.numpy())
plt.plot(x_n, data.numpy())
plt.show()

plt.plot(loss_dyn)
plt.show()
