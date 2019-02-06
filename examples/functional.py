import matplotlib.pyplot as plt
import numpy as np
from wave1D import functional
from wave1D import signal_processing

T = 10.0
n = 1500
dt = T / n

t = np.linspace(0., T, n)
u = functional.ricker(t - 2.0, 2.0)
uhat, freqs = signal_processing.frequency_synthesis(u, T, dt)
v = np.fft.irfft(uhat)

plt.subplot(211)
plt.plot(t, u)
plt.plot(t, v)

plt.subplot(212)
plt.plot(freqs, uhat)
plt.xlim([0.0, 10.0])
plt.show()
