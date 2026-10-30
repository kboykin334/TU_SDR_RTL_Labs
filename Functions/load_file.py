import numpy as np 
from pathlib import Path 

def load_file(filename):
    """
    Read interleaved unisgned 8-bit RTL-SDR I/Q samples 
    and return complex samples -- 
    """
    
    samples = np.fromfile(filename, dtype=np.uint8).astype(np.float64)

    #Extra syntax to ensure complete I/Q pairs 
    if len(samples) % 2 !=0:
        samples = samples[:-1]

    # centers unsigned 8-bit RTL-SDR values 
    samples = samples - 127 

    # forms complex I/Q vector 
    samples = samples[0::2] + 1j * samples[1::2] 

    return samples