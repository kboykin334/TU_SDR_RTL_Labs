import numpy as np


def ffreq(x, fs, fr, dt, sq):
    """
    Find signal frequency offsets in raw RTL-SDR IQ data.

    x: data 
    fs: sample rate 
    fr: frequency resolution 
    dt: time sampling period 
    sq: squelch treshold 
    """

    x = np.asarray(x)


    nf = int(fs / fr) # FFT frame size 
    nt = int(fs * dt) # samples btwn FFT frames 

    spectra = []

    # Compute sparse FFTs
    for start in range(0, len(x) - nf + 1, nt):
        block = x[start : start + nf]

        spectrum = np.fft.fftshift(
            np.fft.fft(block)
        )

        spectra.append(spectrum)

    if not spectra:
        return np.array([])

    xf = np.vstack(spectra) # stacks FFT frames 

    xfm = np.max(np.abs(xf), axis=0) # max magnitude at each freq 

    mx = np.max(xfm) # overall max 

    if mx == 0:
        return np.array([])

    # Actual FFT frequency offsets
    frequency_axis = np.fft.fftshift(
        np.fft.fftfreq(nf, d=1 / fs)
    )

    detected = xfm > (sq * mx) # squelch treshold 

    return frequency_axis[detected]