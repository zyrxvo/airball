import numpy as _numpy

def moving_average(a, n=3) :
    ret = _numpy.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n