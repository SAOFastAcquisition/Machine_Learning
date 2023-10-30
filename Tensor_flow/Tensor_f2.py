import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import math


def quiet_sun(_x, _level=1.0, _edge=90, _width=10):
    _y1 = (1 + tf.math.erf((_x + _edge) / _width)) / 2
    _y2 = -(1 + tf.math.erf((_x - _edge) / _width)) / 2
    return (_y1 + _y2) * _level


x = tf.Variable([[-2.0, 0.0, 2.0]])
edge = tf.Variable(10.0)
width = tf.Variable(3.0)
with tf.GradientTape() as tape:
    # y = (1.0 + math.erf((x + edge) / width)) / 2.0 - (1.0 + math.erf((x - edge) / width)) / 2.0
    # y = quiet_sun(x, 1.0, edge, width)
    #
    y = tf.math.exp(-(x - edge) ** 2.0 / width)
df = tape.gradient(y, [edge, width])
print(df[0], df[1], sep='\n')



