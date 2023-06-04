import numpy as np


def get_ecdf(sample, weights=None):
    """
    Get empirical CDF from a weighted 1D sample
    """

    assert len(sample.shape) == 1, "Only 1D CDF is implemented"

    if weights is None:
        weights = np.ones_like(sample, dtype=sample.dtype)
    assert sample.shape == weights.shape

    i = np.argsort(sample)
    x, w = sample[i], weights[i]

    w_cumsum = np.cumsum(w)
    assert w_cumsum[-1] > 0

    w_cumsum /= w_cumsum[-1]
    return x, w_cumsum


def _interleave_ecdfs(x1, y1, x2, y2):
    """
    Interleave two eCDFs by their argument
    """
    assert len(x1.shape) == len(x2.shape) == 1
    assert x1.shape == y1.shape
    assert x2.shape == y2.shape

    x = np.sort(np.concatenate([x1, x2]))
    y1 = np.insert(y1, 0, [0])
    y2 = np.insert(y2, 0, [0])
    return (
        x,
        y1[np.searchsorted(x1, x, side="right")],
        y2[np.searchsorted(x2, x, side="right")],
    )


def ks_2samp_w(data1, data2, w1=None, w2=None):
    cdf1 = get_ecdf(data1, w1)
    cdf2 = get_ecdf(data2, w2)
    _, cdf1_i, cdf2_i = _interleave_ecdfs(*cdf1, *cdf2)
    return np.abs(cdf2_i - cdf1_i).max()
