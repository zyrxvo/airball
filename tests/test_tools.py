import airball
import airball.units as u
import numpy as np


def test_isList():
    arr1 = [1, 2, 3]
    arr2 = np.array([1, 2, 3])
    arr3 = [1, 2, 3] << u.km
    val1 = 1
    val2 = np.array([1])
    val3 = 1 << u.km
    assert airball.tools.isList(arr1)
    assert airball.tools.isList(arr2)
    assert airball.tools.isList(arr3)
    assert not airball.tools.isList(val1)
    assert airball.tools.isList(val2)  # an ndarray is always a list
    assert not airball.tools.isList(val3)  # a single Quanity is not a list
