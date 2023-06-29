import pytest
import airball
import airball.units as u
import numpy as np

def test_initialize_Stars_with_lists():
    stars = airball.Stars(m=[1,2,3], b=[4,5,6], v=[7,8,9])
    assert np.all(stars.m == np.array([1,2,3]) * u.solMass)
    assert np.all(stars.b == np.array([4,5,6]) * u.au)
    assert np.all(stars.v == np.array([7,8,9]) * u.km/u.s)

def test_initialize_Stars_with_ndarrays():
    stars = airball.Stars(m=np.array([1,2,3]), b=np.array([4,5,6]), v=np.array([7,8,9]))
    assert np.all(stars.m == np.array([1,2,3]) * u.solMass)
    assert np.all(stars.b == np.array([4,5,6]) * u.au)
    assert np.all(stars.v == np.array([7,8,9]) * u.km/u.s)

def test_initialize_Stars_with_lists_and_units():
    stars = airball.Stars(m=[1,2,3]*u.solMass, b=[4,5,6]*u.au, v=[7,8,9]*u.km/u.s)
    assert np.all(stars.m == np.array([1,2,3]) * u.solMass)
    assert np.all(stars.b == np.array([4,5,6]) * u.au)
    assert np.all(stars.v == np.array([7,8,9]) * u.km/u.s)

def test_initialize_Stars_with_ndarrays_and_units():
    stars = airball.Stars(m=np.array([1,2,3])*u.solMass, b=np.array([4,5,6])*u.au, v=np.array([7,8,9])*u.km/u.s)
    assert np.all(stars.m == np.array([1,2,3]) * u.solMass)
    assert np.all(stars.b == np.array([4,5,6]) * u.au)
    assert np.all(stars.v == np.array([7,8,9]) * u.km/u.s)

def test_initialize_Stars_with_mixed_lists():
    stars = airball.Stars(m=[1,2,3], b=np.array([4,5,6]), v=[7,8,9]*u.km/u.s)
    assert np.all(stars.m == np.array([1,2,3]) * u.solMass)
    assert np.all(stars.b == np.array([4,5,6]) * u.au)
    assert np.all(stars.v == np.array([7,8,9]) * u.km/u.s)

def test_initialize_Stars_with_mixed_lists_and_floats():
    stars = airball.Stars(m=1, b=np.array([4,5,6]), v=7*u.km/u.s)
    assert np.all(stars.m == np.array([1,1,1]) * u.solMass)
    assert np.all(stars.b == np.array([4,5,6]) * u.au)
    assert np.all(stars.v == np.array([7,7,7]) * u.km/u.s)

def test_initialize_Stars_with_floats_and_size():
    stars = airball.Stars(m=1, b=3.0, v=7*u.km/u.s, size=3)
    assert np.all(stars.m == np.array([1,1,1]) * u.solMass)
    assert np.all(stars.b == np.array([3,3,3]) * u.au)
    assert np.all(stars.v == np.array([7,7,7]) * u.km/u.s)

def test_initialize_Stars_with_uneven_lists():
    with pytest.raises(airball.ListLengthException):
        airball.Stars(m=[1], b=[4,5,6], v=[7,8,9])
    with pytest.raises(airball.ListLengthException):
        airball.Stars(m=[1,2,3], b=[4], v=[7,8,9])
    with pytest.raises(airball.ListLengthException):
        airball.Stars(m=[1,2,3], b=[4,5,6], v=[7])

def test_initialize_Stars_with_unspecified_parameters():
    with pytest.raises(airball.UnspecifiedParameterException):
        airball.Stars(b=[4,5,6], v=[7,8,9])
    with pytest.raises(airball.UnspecifiedParameterException):
        airball.Stars(m=[1,2,3], v=[7,8,9])
    with pytest.raises(airball.UnspecifiedParameterException):
        airball.Stars(m=[1,2,3], b=[4,5,6])

def test_initialize_Stars_with_floats_and_unspecified_size():
    with pytest.raises(airball.UnspecifiedParameterException):
        airball.Stars(m=1, b=3, v=7)

def test_initialize_Stars_with_lists_and_specified_size():
    with pytest.raises(airball.OverspecifiedParametersException):
        airball.Stars(m=[1,2], v=[7,8,9], size=4)

def test_initialize_Stars_with_custom_object():
    # Test using a custom object that defines __len__.
    class CustomObject:
        def __init__(self, data): self.data = data
        def __len__(self): return len(self.data)

    # Create an instance of the custom object
    my_object = CustomObject([1, 2, 3])
    with pytest.raises(airball.IncompatibleListException):
        airball.Stars(m=my_object, b=[4,5,6], v=[7,8,9])
        airball.Stars(m=[1,2,3], b=my_object, v=[7,8,9])
        airball.Stars(m=[1,2,3], b=[4,5,6], v=my_object)

