import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def quiet_sun(_x, _level=1.0, _edge_l=-90, _edge_r=90, _width_l=10, _width_r=10):
    _y1 = (1 + tf.math.erf((_x - _edge_l) / _width_l)) / 2
    _y2 = -(1 + tf.math.erf((_x - _edge_r) / _width_r)) / 2
    return (_y1 + _y2) * _level


def flare(_x, _height, _pos, _width):
    # _len = tf.size(_pos).numpy()
    _len = tf.size(_pos)
    _flare_sum = tf.Variable(tf.zeros_like(_x, dtype=float))
    for i in range(_len):
        _flare = _height[i] * tf.exp(-0.5 * ((_x - _pos[i]) / _width[i]) ** 2)
        _flare_sum = _flare_sum + _flare
    return _flare_sum


def sun_with_noise(_x, _level=1.0, _edge_l=-90, _edge_r=90, _width_l=10, _width_r=10):
    _len = int(tf.size(_x).numpy())
    _sun = quiet_sun(_x, _level, _edge_l, _edge_r, _width_l, _width_r)
    _noise = tf.random.normal(shape=[_len], stddev=0.01)
    return _sun + _noise


def model():
    _end = 30.0
    _step_x = 0.1
    _x_n = np.arange(-_end, _end + 0.1, _step_x)
    _data = tf.math.log(sun_with_noise(tf.Variable(_x_n, dtype=float), 1.0, -19.0, 21.0, 2.0, 2.5) +
                        flare(tf.Variable(_x_n, dtype=float), [1.5, 2.5, 4], [3.0, -5.0, 0], [1.0, 1.8, 1.5]) + 1.0)
    return _x_n, _data


@tf.function
def batch_train(_x_batch, _y_batch):

    _y_batch = data[n_batch * BATCH_SIZE: (n_batch + 1) * BATCH_SIZE]
    _x_batch = x_n[n_batch * BATCH_SIZE: (n_batch + 1) * BATCH_SIZE]

    with tf.GradientTape() as t:
        f = quiet_sun(_x_batch, level_mod, edge_mod_l, edge_mod_r, width_mod_l, width_mod_r) + \
            flare(_x_batch, height_mod1, pos_mod1, width1_mod1) + b
        d_loss = tf.reduce_mean(tf.square(_y_batch - f))

    dp1, dp2, dp3, dp4, dp5, db, dh, dp, dw = t.gradient(d_loss,
                                                         [level_mod, edge_mod_l, edge_mod_r, width_mod_l,
                                                          width_mod_r, b,
                                                          height_mod1, pos_mod1, width1_mod1])
    opt.apply_gradients(zip([dp1, dp2, dp3, dp4, dp5, db, dh, dp, dw],
                            [level_mod, edge_mod_l, edge_mod_r, width_mod_l, width_mod_r, b,
                             height_mod1, pos_mod1, width1_mod1]))

    return d_loss, dp1, dp2, dp3, dp4, dp5, db, dh, dp, dw


mod = 'n'
#                   *** Вызов модельных данных ***
if mod == 'y':
    x_n, data = model()
#                   *** Загрузка реальных данных ***
else:
    path1 = Path("2023-02-17_04+16" + '_scan_freq.npy')
    path2 = Path("2023-02-17_04+16" + '_time.npy')
    dat = np.load(path1, allow_pickle=True)[6, 170:620]
    data = tf.constant(dat, dtype=float)

    x_n = np.load(path2, allow_pickle=True)[170:620]
# fig, axes = plt.subplots()
# axes.semilogy(x_n, dat)
# axes.grid(True)
# plt.show()
#                   *** Model ***
level_mod = tf.Variable(5500.0, name='sun_level')
edge_mod_l = tf.Variable(150.0, name='left_edge')
edge_mod_r = tf.Variable(290.0, name='right_edge')
width_mod_l = tf.Variable(7.0, name='left_slope')
width_mod_r = tf.Variable(7.0, name='right_slope')
b = tf.Variable(500.0, name='ground')

height_mod1 = tf.Variable([4000.0, 5000.0, 9000.0, 14000.0, 3000.0, 3000.0, 5000.0, 1000], name='height1')
pos_mod1 = tf.Variable([161.0, 179.0, 191.0, 200.0, 222.0, 255.0, 266.0, 280], name='pos1')
width1_mod1 = tf.Variable([7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0], name='width1')

opt = tf.optimizers.Adam(learning_rate=0.5)  # EPOCHS = 80

EPOCHS = 100
# learnin_rate = 3.0
loss_dyn = np.array([])

BATCH_SIZE = 225
TOTAL_POINTS = len(x_n)
num_steps = TOTAL_POINTS // BATCH_SIZE

for n in range(EPOCHS):
    for n_batch in range(num_steps):
        y_batch = data[n_batch * BATCH_SIZE: (n_batch + 1) * BATCH_SIZE]
        x_batch = x_n[n_batch * BATCH_SIZE: (n_batch + 1) * BATCH_SIZE]

        with tf.GradientTape() as t:
            f = quiet_sun(x_batch, level_mod, edge_mod_l, edge_mod_r, width_mod_l, width_mod_r) + \
                flare(x_batch, height_mod1, pos_mod1, width1_mod1) + b
            d_loss = tf.reduce_mean(tf.square(y_batch - f))

        dp1, dp2, dp3, dp4, dp5, db, dh, dp, dw = t.gradient(d_loss,
                                                              [level_mod, edge_mod_l, edge_mod_r, width_mod_l,
                                                               width_mod_r, b,
                                                               height_mod1, pos_mod1, width1_mod1])
        opt.apply_gradients(zip([dp1, dp2, dp3, dp4, dp5, db, dh, dp, dw],
                                [level_mod, edge_mod_l, edge_mod_r, width_mod_l, width_mod_r, b,
                                 height_mod1, pos_mod1, width1_mod1]))
        loss_dyn = np.append(loss_dyn, d_loss.numpy())
    n_batch = 0

sun_calc = quiet_sun(x_n, level_mod, edge_mod_l, edge_mod_r, width_mod_l, width_mod_r) + \
           flare(x_n, height_mod1, pos_mod1, width1_mod1) + b
sun_quiet = quiet_sun(x_n, level_mod, edge_mod_l, edge_mod_r, width_mod_l, width_mod_r) + b
data_n = dat - quiet_sun(x_n, level_mod, edge_mod_l, edge_mod_r, width_mod_l, width_mod_r)
print(level_mod, edge_mod_l, edge_mod_r, width_mod_l, width_mod_r, b, height_mod1, pos_mod1, width1_mod1, sep='\n')
fig, axes = plt.subplots()
axes.semilogy(x_n, sun_calc.numpy())
axes.semilogy(x_n, sun_quiet.numpy())
axes.semilogy(x_n, dat)
axes.semilogy(x_n, data_n.numpy())
axes.grid(True)
plt.show()

plt.plot(loss_dyn)
plt.show()

height_mod2 = tf.Variable(height_mod1, name='height')
pos_mod2 = tf.Variable(pos_mod1, name='pos')
width1_mod2 = tf.Variable(width1_mod1, name='width')

opt = tf.optimizers.Adam(learning_rate=2.5)  # EPOCHS = 80

loss_dyn = np.array([])
EPOCHS = 300
BATCH_SIZE = 450
TOTAL_POINTS = len(x_n)
num_steps = TOTAL_POINTS // BATCH_SIZE

for n in range(EPOCHS):
    loss = tf.Variable(0.0)
    for n_batch in range(num_steps):
        y_batch = data_n[n_batch * BATCH_SIZE: (n_batch + 1) * BATCH_SIZE]
        x_batch = x_n[n_batch * BATCH_SIZE: (n_batch + 1) * BATCH_SIZE]
        with tf.GradientTape() as t:
            f = flare(x_batch, height_mod2, pos_mod2, width1_mod2) + b
            d_loss = tf.reduce_mean(tf.square(y_batch - f))
        loss.assign(d_loss)
        loss_dyn = np.append(loss_dyn, loss.numpy())
        dh1, dp1, dw1, db1 = t.gradient(d_loss, [height_mod2, pos_mod2, width1_mod2, b])
        opt.apply_gradients(zip([dh1, dp1, dw1, db1],
                                [height_mod2, pos_mod2, width1_mod2, b]))
    n_batch = 0
sun_calc1 = flare(x_n, height_mod2, pos_mod2, width1_mod2) + b

dat2 = dat - sun_calc1.numpy()

print(b, height_mod2, pos_mod2, width1_mod2, sep='\n')
plt.plot(x_n, sun_calc1.numpy())
plt.plot(x_n, data_n.numpy())
plt.plot(x_n, dat2)
plt.show()

plt.plot(loss_dyn)
plt.show()
