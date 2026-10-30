import numpy as np


def decode_id(pd):
    dcd = "#ABCDEFGHIJKLMNOPQRSTUVWXYZ#####_###############0123456789######"

    pd = np.atleast_2d(pd)
    npackets, lp = pd.shape

    if lp < 82:
        raise ValueError("Each packet must contain at least 82 bits.")

    for jj in range(npackets):
        cs = ""

        for kk in range(0, 42, 6):
            bits = pd[jj, kk + 40 : kk + 46]
            nc = int(np.sum(bits * (2 ** np.arange(5, -1, -1))))
            cs += dcd[nc]

        print(cs)




def hex2bin(s):
    bits = []

    for char in s:
        value = int(char, 16)
        bits.extend(bta[value])

    return np.array(bits, dtype=int)