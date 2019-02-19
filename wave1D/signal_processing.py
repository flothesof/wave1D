import numpy as np


def frequency_synthesis(values, timestep, n_fft=None):
    """
    Performing fourier transform of an input time signal.
    :param values: discrete values of a time functional.
    :param timestep: time step associated to the input data.
    :param n_fft: (optional) number of point used in FFT transform.
    :return: the fourrier analysis of a real signal and the associated frequencies.
    """
    if timestep < 0.0:
        raise ValueError()

    if n_fft is None:
        nvalues = len(values)
    else:
        nvalues = np.max([len(values), n_fft])

    freqs = np.fft.rfftfreq(n=nvalues, d=timestep)
    amp = np.fft.rfft(values, n=nvalues) * (timestep / nvalues)

    return amp, freqs


def apply_attenuation_filter(values, timestep, path_length=0.0, attenuation_filter=lambda f: 0.0):
    """
    Applying attenuation filter on the frequency synthesis of an input signal.
    :param values: discrete values of a time functional.
    :param timestep: time step associated to the input data.
    :param path_length: travelled distance.
    :param attenuation_filter: attenuation values depending on the frequency (WARNING: not the angular frequency).
    :return:
    """
    hat_values, freqs = frequency_synthesis(values, timestep)
    attenuated_hat_values = np.array([hat_values[i] * np.exp(-attenuation_filter(f) * path_length) for i, f in enumerate(freqs)])
    return np.fft.irfft(attenuated_hat_values)


