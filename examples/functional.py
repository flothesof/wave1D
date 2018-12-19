import matplotlib.pyplot as plt
import numpy as np
from wave1D import functional


x = np.linspace(0., 30., 1500)
plt.plot(x, functional.ricker(x - 10.0, 0.1))
plt.show()