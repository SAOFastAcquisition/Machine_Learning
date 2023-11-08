
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

flares_n = np.array([])
x_t = tf.Variable(-30, dtype=float)
while x_t < 30.1:
    flares = flare(x_t, height, pos, f_width) + quiet_sun(x_t, level, edge, width)
    flares_n = np.append(flares_n, flares.numpy())
    x_t.assign_add(0.1)

plt.plot(x_n, flares_n)
plt.show()

# level_mod.assign_sub(learning_rate * dp1)
        # edge_mod.assign_sub(learning_rate * dp2)
        # width_mod.assign_sub(learning_rate * dp3)
        # height_mod.assign_sub(learning_rate * dp4)
        # pos_mod.assign_sub(learning_rate * dp5)
        # width1_mod.assign_sub(learning_rate * dp6)

# opt = tf.optimizers.SGD(learning_rate=3.0)                  # Классический стохастический градиентный метод
# opt = tf.optimizers.SGD(momentum=0.1, learning_rate=3.0)    # Метод моментов  EPOCHS = 120
# opt = tf.optimizers.SGD(momentum=0.1, nesterov=True, learning_rate=3.0)    # Метод Нестерова  EPOCHS = 120
# opt = tf.optimizers.Adagrad(learning_rate=3.0)              # EPOCHS = 120
# opt = tf.optimizers.Adadelta(learning_rate=50.0)            # EPOCHS = 60
# opt = tf.optimizers.RMSprop(learning_rate=0.02)            # EPOCHS = 120