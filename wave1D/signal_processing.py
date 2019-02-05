import numpy as np


def frequency_synthesis(values, max_time, timestep, time_origin=0.0):
    """
    Performing fourier transform of an input time signal.
    :param values: discrete values of a time functional.
    :param max_time: maximal time window.
    :param timestep: time step associated to the input data.
    :param time_origin: potential time origin.
    :return: the fourrier analysis of a real signal and the associated frequencies.
    """
    if max_time < time_origin:
        raise ValueError()
    if timestep < 0.0:
        raise ValueError()

    time_window = (max_time - time_origin)
    nvalues = len(values)
    frequency_step = 1.0 / time_window
    return np.abs(np.fft.rfft(values)), np.array([frequency_step * i for i in range(int(nvalues/2) + 1)])
