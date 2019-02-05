import matplotlib.pyplot as plt
import numpy as np
from wave1D import functional


x = np.linspace(0., 10., 1500)
u = functional.ricker(x - 2.0, 2.0)

#plt.plot(x, u)
#plt.show()

ufft = np.fft.rfft(u)
plt.plot(ufft)