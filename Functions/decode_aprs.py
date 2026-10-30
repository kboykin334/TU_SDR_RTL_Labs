import numpy as np

# 1. Decode NRZI
def decode_NRZI(bts):
    """
    Decode an NRZI bit stream.
    Same consecutive input bits -> 1
    Different consecutive input bits -> 0
    Initial previous input bit is 0.
    """
    bts = np.asarray(bts, dtype=int).flatten()
    db = np.zeros(len(bts), dtype=int)

    previous = 0

    for i, bit in enumerate(bts):
        db[i] = 1 if bit == previous else 0
        previous = bit

    return db


# 2. Synchronize to APRS frame flags
def frame_byte_sync(db):
    """
    Find the first 01111110 frame flag.
    Skip consecutive flag bytes and return the bits
    following the last leading flag.
    """
    db = np.asarray(db, dtype=int).flatten()
    flag = np.array([0, 1, 1, 1, 1, 1, 1, 0])

    # Find the first flag, checking one bit at a time
    first = -1

    for i in range(len(db) - 7):
        if np.array_equal(db[i:i+8], flag):
            first = i
            break

    if first == -1:
        print("No APRS frame flag found.")
        return np.array([], dtype=int)

    print("First frame flag starts at bit:", first)

    # Skip consecutive flags, one byte at a time
    pos = first + 8
    flags_found = 1

    while pos + 8 <= len(db):
        if np.array_equal(db[pos:pos+8], flag):
            flags_found += 1
            pos += 8
        else:
            break

    print("Leading frame flags found:", flags_found)

    return db[pos:]


# 3. Remove bit-stuffed zeros
def de_bit_stuff(dbs):
    """
    Remove a zero inserted after five consecutive ones.
    """
    dbs = np.asarray(dbs, dtype=int).flatten()
    output = []
    ones_count = 0
    i = 0

    while i < len(dbs):
        bit = dbs[i]

        # After five ones, discard the next zero if present
        if ones_count == 5 and bit == 0:
            ones_count = 0
            i += 1
            continue

        output.append(bit)

        if bit == 1:
            ones_count += 1
        else:
            ones_count = 0

        i += 1

    return np.array(output, dtype=int)


# 4. Convert bits into bytes and ASCII characters
def bits_to_char(b):
    """
    Take a decoded bit stream and convert it to characters.
    Bits are transmitted least-significant bit first.
    """
    b = np.asarray(b, dtype=int).flatten()

    # Use only complete groups of 8 bits
    num_bytes = len(b) // 8
    b = b[:num_bytes * 8]

    cs = ""

    for i in range(num_bytes):
        byte = b[i*8:(i+1)*8]

        # Convert 8 bits to a byte value, LSB first
        bv = sum(byte[j] * (2**j) for j in range(8))

        cs += chr(bv)

    return cs