import numpy as np
import matplotlib.pyplot as plt

lambda1 = np.arange(0, 5.0 + 0.1, 0.1)
lambda2 = np.arange(0, 5.0 + 0.1, 0.1)
L1, L2 = np.meshgrid(lambda1, lambda2)

k = 0.2
c = L1 * L2 - k * (L1 + L2) ** 2

plt.figure()
plt.rcParams.update({"font.size": 15})
C = plt.contour(L1, L2, c)
plt.clabel(C, inline=True)
plt.xlabel("lambda_1")
plt.ylabel("lambda_2")
plt.title(f"Contour of C(x, y) for k = {k}")
plt.axis("equal")
plt.axis([0, 5, 0, 5])
plt.show()
