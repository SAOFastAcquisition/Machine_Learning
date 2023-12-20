import matplotlib.pyplot as plt
import numpy as np


ax1 = plt.subplot(1, 2, 1)
plt.plot(np.random.random(10))
ax2 = plt.subplot(1, 3, 3)
plt.plot(np.random.random(10))
plt.show()