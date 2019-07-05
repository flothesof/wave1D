import matplotlib.pyplot as plt
import numpy as np
from wave1D import functional

xmax = 2.0
coef = 666.0
x = np.linspace(-5.0, 5.0, 1000)
f = coef * np.ones_like(x) - coef * (functional.heaviside(x + xmax, 0.1) - functional.heaviside(x - xmax, 0.1))
plt.plot(x, f)
plt.show()