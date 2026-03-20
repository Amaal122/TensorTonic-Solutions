import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p (drop probability).
    Return (output, dropout_pattern).
    """
    if rng is None:
        rng = np.random.random(123)

    # create mask (1 = keep, 0 = drop)
    x = np.array(x)
    mask = rng.random(x.shape) > p

    # scale output
    out = (x * mask) / (1 - p)

    return out, (mask/(1-p))