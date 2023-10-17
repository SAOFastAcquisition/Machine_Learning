import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0, 10.1, 0.1)
y = 1 / (1 + 10 * np.square(x))

xx = x[:: 2]
yy = y[:: 2]
xx_c = x[1:: 2]
yy_c = y[1:: 2]

z_train = np.polyfit(xx, yy, 50)
print(z_train)
p = np.poly1d(z_train)
y_fit = p(x)

plt.scatter(xx, yy, color='red')
plt.scatter(xx_c, yy_c, color='blue')
plt.plot(x, y_fit, color='green')

# plt.xlim([0, 45])
# plt.ylim([0, 75])
# plt.ylabel("длина")
# plt.xlabel("ширина")
plt.grid(True)
plt.show()
pass