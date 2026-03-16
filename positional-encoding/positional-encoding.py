import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):

    M = np.zeros((seq_len, d_model))

    for pos in range(seq_len):
        for i in range(d_model // 2):
            angle = pos / (base ** (2*i / d_model))
            M[pos, 2*i] = np.sin(angle)
            M[pos, 2*i+1] = np.cos(angle)

        if d_model % 2 == 1:
            i = d_model // 2
            angle = pos / (base ** (2*i / d_model))
            M[pos, -1] = np.sin(angle)

    return M