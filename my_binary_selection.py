import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([[10, 50], [20, 30], [25, 30], [20, 60], [15, 70], [40, 40], [30, 45], [20, 45], [40, 30], [7, 35]])
y_train = np.array([-1, 1, 1, -1, -1, 1, 1, -1, 1, -1])

step = 0.05
n_step = 50
eps = 0.01
vector = [0, -1]

len_train = len(y_train)

for i in range(n_step):
    for j in range(len_train):
        M = (x_train[j][0] * (1 - eps * y_train[j]) * vector[0] + x_train[j][1] * vector[1]) * y_train[j]    # Отступ
        if M < 0:
            vector[0] += step * y_train[j]

    Q = sum([1 for j in range(len_train) if np.dot(x_train[j], vector) * y_train[j] < 0])   # Функция ошибок (показатель качества)
    if Q == 0:
        break

print(f'vector = {vector}')
line_x = list(range(max(x_train[:, 0])))    # формирование графика разделяющей линии
line_y = [vector[0]*x for x in line_x]

x_0 = x_train[y_train == 1]                 # формирование точек для 1-го
x_1 = x_train[y_train == -1]                # и 2-го классов

plt.scatter(x_0[:, 0], x_0[:, 1], color='red')
plt.scatter(x_1[:, 0], x_1[:, 1], color='blue')
plt.plot(line_x, line_y, color='green')

plt.xlim([0, 45])
plt.ylim([0, 75])
plt.ylabel("длина")
plt.xlabel("ширина")
plt.grid(True)
plt.show()