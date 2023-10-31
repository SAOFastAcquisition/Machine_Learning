import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

a = np.array([a for a in range(30)])
b = np.array([a * 2 for a in range(1, 10)])

a_t = tf.constant(a, dtype=float)
b_t = tf.Variable(b, dtype=float)

a_r = tf.reshape(a_t, [2, 5, 3])
# print(a_r)
a_r1 = tf.reshape(a_r, [5, 6])
# print(f'a_t = {a_t} \na_r = {a_r} \na_r1 = {a_r1}')

b_tr = tf.reshape(b_t, [3, 3])
# print(f'b_tr = {b_tr}')
b_T = tf.transpose(b_tr, [1, 0])
# print(f'b_T = {b_T}')
b_ones = tf.ones_like(b_T)
b_zeros = tf.zeros_like(b_ones)
print(f'b_ones = {b_ones} \nb_zeros = {b_zeros}')
f_t = tf.fill([3, 3], 5)
print(f'f_t = {f_t}')
b_iden = tf.identity(b_tr)

print(b_tr)
b_tr_sh = tf.random.shuffle(b_tr)

print(b_tr_sh)
# print(f'b_iden = {b_iden} \nb_tr = {b_tr}')





