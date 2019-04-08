import matplotlib.pyplot as plt
import numpy as np
from wave1D import functional
from wave1D import signal_processing

T = 10.0
n = 1500
dt = T / n

t = np.linspace(0., T, n)
u = functional.hanning(t, 4.0, 6)
uhat, freqs = signal_processing.frequency_synthesis(u, dt)
v = np.fft.irfft(uhat)

plt.subplot(211)
plt.plot(t, u / np.max(u))
plt.plot(t, v / np.max(v))

plt.subplot(212)
plt.plot(freqs, np.abs(uhat))
plt.xlim([0.0, 10.0])
plt.show()
