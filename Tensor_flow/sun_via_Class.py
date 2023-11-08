import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np


class DenseNN(tf.Module):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs
        self.fl_init = False

    def __call__(self, _x):
        if not self.fl_init:
            self.fl_init = True
            self.sun = tf.Variable([5000.0, 140.0, 270.0, 15.0, 15.0], dtype=float, name='sun')
            self.bias = tf.Variable(500.0, dtype=float, name='bias')

            self.height_f = tf.Variable([4000.0, 5000.0, 9000.0, 14000.0, 3000.0, 3000.0, 5000.0, 1000],
                                        dtype=tf.float32, name="height")
            self.pos_f = tf.Variable([161.0, 179.0, 191.0, 200.0, 222.0, 255.0, 266.0, 280], dtype=float, name='pos_f')
            self.width_f = tf.Variable([7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0], dtype=float, name='width_f')

        _y1 = (1 + tf.math.erf((_x - self.sun[1]) / self.sun[3])) / 2
        _y2 = -(1 + tf.math.erf((_x - self.sun[2]) / self.sun[4])) / 2
        _y = self.sun[0] * (_y1 + _y2)
        return _y


a = DenseNN(1)
s = a.trainable_variables
print(a.fl_init)
x = np.arange(140.0, 280.0 + 0.1, 0.2)
y = a(x)
print(y)
