from lib2to3.pytree import convert
import numpy as _numpy

convert_kms_to_auyr2pi = 0.03357365989646266 # 1 km/s to AU/Yr2Pi
convert_auyr2pi_to_kms = 1/convert_kms_to_auyr2pi

def moving_average(a, n=3) :
    ret = _numpy.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n