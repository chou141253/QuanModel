import matplotlib.pyplot as plt
import time
import numpy as np

"""
with n_scale : 1
100%|##########################################################################| 10000/10000 [02:44<00:00, 60.70it/s]
test accuracy: 9033/10000 = 90.32999420166016%

with n_scale : 2
100%|##########################################################################| 10000/10000 [02:45<00:00, 60.45it/s]
test accuracy: 9018/10000 = 90.18000030517578%

with n_scale : 3
100%|##########################################################################| 10000/10000 [02:45<00:00, 60.57it/s]
test accuracy: 8649/10000 = 86.48999786376953%

with n_scale : 4
100%|##########################################################################| 10000/10000 [02:45<00:00, 60.26it/s]
test accuracy: 8365/10000 = 83.6500015258789%

with n_scale : 5
100%|##########################################################################| 10000/10000 [02:45<00:00, 60.55it/s]
test accuracy: 6974/10000 = 69.73999786376953%

with n_scale : 6
100%|##########################################################################| 10000/10000 [02:49<00:00, 59.15it/s]
test accuracy: 5488/10000 = 54.87999725341797%

with n_scale : 7
100%|##########################################################################| 10000/10000 [02:47<00:00, 59.63it/s]
test accuracy: 4063/10000 = 40.62999725341797%

with n_scale : 8
100%|##########################################################################| 10000/10000 [02:48<00:00, 59.50it/s]
test accuracy: 3255/10000 = 32.54999923706055%

with n_scale : 9
100%|##########################################################################| 10000/10000 [02:48<00:00, 59.26it/s]
test accuracy: 1618/10000 = 16.18000030517578%

with n_scale : 10
100%|##########################################################################| 10000/10000 [02:45<00:00, 60.36it/s]
test accuracy: 1146/10000 = 11.460000038146973%

with n_scale : 11
100%|##########################################################################| 10000/10000 [02:43<00:00, 61.01it/s]
test accuracy: 1031/10000 = 10.309999465942383%

with n_scale : 12
100%|##########################################################################| 10000/10000 [02:43<00:00, 61.01it/s]
test accuracy: 1002/10000 = 10.019999504089355%

"""

acc = [90.329, 90.180, 86.489, 83.650, 69.739, 54.879, 40.629, 32.549, 16.180, 11.460, 10.309, 10.019]
scale = [   1,      2,      3,      4,      5,      6,      7,      8,      9,     10,     11,     12] # clock比例

show_id = [0,1,2,5,8]

plt.plot(scale, acc, "bo-", label="4bits")
plt.xlabel("scale")
plt.xticks(np.linspace(1,12,12))
plt.ylabel("accuracy")
plt.yticks(np.linspace(0,100, 10))
plt.ylim(0,100)

for index in show_id:
	plt.text(scale[index], acc[index], str(round(acc[index], 2)))

plt.legend()

plt.grid(True)
plt.savefig('acc.png', dpi=100)
plt.show()
