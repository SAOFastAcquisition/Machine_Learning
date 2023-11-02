import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path


def quiet_sun(_x, _level=1.0, _edge_l=-90, _edge_r=90, _width_l=10, _width_r=10):
    _y1 = (1 + tf.math.erf((_x - _edge_l) / _width_l)) / 2
    _y2 = -(1 + tf.math.erf((_x - _edge_r) / _width_r)) / 2
    return (_y1 + _y2) * _level


def flare(_x, _height, _pos, _width):
    _len = tf.size(_pos).numpy()
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


mod = 'n'
#                   *** Вызов модельных данных ***
if mod == 'y':
    x_n, data = model()
#                   *** Загрузка реальных данных ***
else:
    path1 = Path("/media/anatoly/Samsung_T5/Fast_Acquisition/2023/Converted_data/2023_02_17sun_conv/2023-02-17_04+16" +
                 '_scan_freq.npy')
    path2 = Path("/media/anatoly/Samsung_T5/Fast_Acquisition/2023/Converted_data/2023_02_17sun_conv/2023-02-17_04+16" +
                 '_time.npy')
    dat = np.load(path1, allow_pickle=True)[0, 170:620]
    data = tf.math.log(tf.constant(dat, dtype=float))

    x_n = np.load(path2, allow_pickle=True)[170:620]
# plt.plot(x_n, data)
# plt.show()
#                   *** Model ***
level_mod = tf.Variable(5000.0)
edge_mod_l = tf.Variable(150.0)
edge_mod_r = tf.Variable(290.0)
width_mod_l = tf.Variable(7.0)
width_mod_r = tf.Variable(7.0)
b = tf.Variable(500.0)

height_mod1 = tf.Variable([4000.0, 5000.0, 9000.0, 14000.0, 3000.0, 3000.0, 5000.0])
pos_mod1 = tf.Variable([161.0, 179.0, 191.0, 200.0, 222.0, 255.0, 266])
width1_mod1 = tf.Variable([7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0])

# opt = tf.optimizers.SGD(learning_rate=3.0)                  # Классический стохастический градиентный метод
# opt = tf.optimizers.SGD(momentum=0.1, learning_rate=3.0)    # Метод моментов  EPOCHS = 120
# opt = tf.optimizers.SGD(momentum=0.1, nesterov=True, learning_rate=3.0)    # Метод Нестерова  EPOCHS = 120
opt = tf.optimizers.Adagrad(learning_rate=3.0)              # EPOCHS = 120
# opt = tf.optimizers.Adadelta(learning_rate=50.0)            # EPOCHS = 60
# opt = tf.optimizers.RMSprop(learning_rate=0.02)            # EPOCHS = 120
# opt = tf.optimizers.Adam(learning_rate=0.04)  # EPOCHS = 80

EPOCHS = 100
# learning_rate = 3.0
loss_dyn = np.array([])

BATCH_SIZE = 90
TOTAL_POINTS = len(x_n)
num_steps = TOTAL_POINTS // BATCH_SIZE

for n in range(EPOCHS):
    for n_batch in range(num_steps):
        y_batch = data[n_batch * BATCH_SIZE: (n_batch + 1) * BATCH_SIZE]
        x_batch = x_n[n_batch * BATCH_SIZE: (n_batch + 1) * BATCH_SIZE]
        with tf.GradientTape() as t:
            f = tf.math.log(quiet_sun(x_batch, level_mod, edge_mod_l, edge_mod_r, width_mod_l, width_mod_r) +
                            flare(x_batch, height_mod1, pos_mod1, width1_mod1) + b)
            loss = tf.reduce_mean(tf.square(y_batch - f))
            loss_dyn = np.append(loss_dyn, loss.numpy())
        dp1, dp2, dp3, dp4, dp5, db, dh, dp, dw = t.gradient(loss,
                                                             [level_mod, edge_mod_l, edge_mod_r, width_mod_l,
                                                              width_mod_r, b,
                                                              height_mod1, pos_mod1, width1_mod1])
        opt.apply_gradients(zip([dp1, dp2, dp3, dp4, dp5, db, dh, dp, dw],
                                [level_mod, edge_mod_l, edge_mod_r, width_mod_l, width_mod_r, b,
                                 height_mod1, pos_mod1, width1_mod1]))
        # level_mod.assign_sub(learning_rate * dp1)
        # edge_mod.assign_sub(learning_rate * dp2)
        # width_mod.assign_sub(learning_rate * dp3)
        # height_mod.assign_sub(learning_rate * dp4)
        # pos_mod.assign_sub(learning_rate * dp5)
        # width1_mod.assign_sub(learning_rate * dp6)
    n_batch = 0

sun_calc = tf.math.log(quiet_sun(x_n, level_mod, edge_mod_l, edge_mod_r, width_mod_l, width_mod_r) +
                       flare(x_n, height_mod1, pos_mod1, width1_mod1) + b)
print(level_mod, edge_mod_l, edge_mod_r, width_mod_l, width_mod_r, b, height_mod1, pos_mod1, width1_mod1, sep='\n')
plt.plot(x_n, sun_calc.numpy())
plt.plot(x_n, data)
plt.show()

plt.plot(loss_dyn)
plt.show()
