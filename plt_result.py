import matplotlib.pyplot as plt
import time
import numpy as np

acc = [89.339, 89.229, 86.689, 86.0199, 77.860, 68.559, 57.169, 48.029, 24.719, 14.289, 11.389, 10]
scale = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] # clock比例

plt.plot(scale, acc, "bo-", label="4bits")
plt.xlabel("scale")
plt.xticks(np.linspace(1,12,12))
plt.ylabel("accuracy")
plt.yticks(np.linspace(0,100, 10))
plt.ylim(0,100)

plt.legend()

plt.grid(True)
plt.savefig('acc.png', dpi=100)
plt.show()
