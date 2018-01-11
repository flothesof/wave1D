import matplotlib.pyplot as plt
import numpy as np
from wave1D import functional


x = np.linspace(0., 1., 150)
plt.plot(x, functional.ricker(x - 0.4, 5.0))
plt.show()