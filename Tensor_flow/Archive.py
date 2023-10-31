
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