import numpy as np 
import matplotlib.pyplot as plt


def msg(x, n0, nf, nt, fs=2048000, dbf=40):
    """
    Compute and display a spectrogram 

    Parameters 
    ----------
    x : complex input signal 
    n0 : first sample to use
    nf : number of samples in each FFT/block size for transform 
    nt : number of time steps (blocks) / FFT blocks 
    fs : sample rate in Hz (default 2048000 Hz = 2.048 MHz)
    dbf : dynamic range of the spectrogram in dB (default 40)

    This extracts a segment of x starting at n0, of length nf*nt
   The image plot is in dB, and autoscaled.  This can look very noisy
   if there aren't any interesting signals present.

   Returns 
   ---------
   ds: complex spectrum
       Rows = frequency bins 
       Columns = time blocks 

   freq: frequency offset for each freq bin in MHz 

   time: time value for each block in seconds

    """

    # Python indexing starts at 0 
    start = n0 - 1

    # Number of samples required 
    total_samples = nf * nt

    # Extract requested portion of signal 
    segment = x[start:start + total_samples]

    if len(segment) < total_samples:
        raise ValueError(
            f"Not enough samples. Need {total_samples}, "
            f"but only {len(segment)} are available. "
        )

    # Divide signal into nt blocks, each containing nf samples 
    blocks = segment.reshape(nt, nf)

    # FFT each block 
    spectrum = np.fft.fft(blocks, axis=1)

    # Put zero frequency in the center 
    spectrum = np.fft.fftshift(spectrum, axes=1)

    # Transpose so rows = frequency and columns = time 
    ds = spectrum.T 

    # Frequency axis
    freq = np.fft.fftshift(
        np.fft.fftfreq(nf, d=1 / fs)
    )

    # Time axis
    time = np.arange(nt) * (nf / fs)

    # Convert magnitude to dB for display 
    magnitude_db = 20 * np.log10(np.abs(ds) + 1e-12)

    # Determine display range 
    max_db = np.max(magnitude_db)
    min_db = max_db - dbf 

    # Display spectrum 
    plt.figure(figsize=(12, 6))

    plt.imshow(
        magnitude_db, 
        aspect="auto",
        origin="lower",
        vmin=min_db, 
        vmax=max_db
    )

    plt.xlabel("Time Block")
    plt.ylabel("Frequency Bin")
    plt.title("Spectrogram")
    plt.colorbar(label="Magnitude (dB)")

    plt.show()

    return ds 
    