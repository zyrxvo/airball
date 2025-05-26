import pytest
import airball
import airball.units as u
import numpy as np

################################################
################################################
##########  INITIALIZATION TESTS  ##############
################################################
################################################


def test_initialize_Stars_with_lists():
    stars = airball.Stars(m=[1, 2, 3], b=[4, 5, 6], v=[7, 8, 9])
    assert np.all(stars.m == np.array([1, 2, 3]) * u.solMass)
    assert np.all(stars.b == np.array([4, 5, 6]) * u.au)
    assert np.all(stars.v == np.array([7, 8, 9]) * u.km / u.s)


def test_initialize_Stars_with_ndarrays():
    stars = airball.Stars(
        m=np.array([1, 2, 3]), b=np.array([4, 5, 6]), v=np.array([7, 8, 9])
    )
    assert np.all(stars.m == np.array([1, 2, 3]) * u.solMass)
    assert np.all(stars.b == np.array([4, 5, 6]) * u.au)
    assert np.all(stars.v == np.array([7, 8, 9]) * u.km / u.s)


def test_initialize_Stars_with_lists_and_units():
    stars = airball.Stars(
        m=[1, 2, 3] * u.solMass, b=[4, 5, 6] * u.au, v=[7, 8, 9] * u.km / u.s
    )
    assert np.all(stars.m == np.array([1, 2, 3]) * u.solMass)
    assert np.all(stars.b == np.array([4, 5, 6]) * u.au)
    assert np.all(stars.v == np.array([7, 8, 9]) * u.km / u.s)

    stars = airball.Stars(
        m=[1 * u.solMass, 2000 * u.jupiterMass, 1e6 * u.earthMass],
        b=[4 * u.au, 5 * u.au, 6 * u.au],
        v=[7 * u.km / u.s, 8 * u.km / u.s, 9 * u.km / u.s],
    )
    assert np.all(
        stars.m == np.array([1, 1.90918846793865, 3.003489348850793]) * u.solMass
    )
    assert np.all(stars.b == np.array([4, 5, 6]) * u.au)
    assert np.all(stars.v == np.array([7, 8, 9]) * u.km / u.s)


def test_initialize_Stars_with_ndarrays_and_units():
    stars = airball.Stars(
        m=np.array([1, 2, 3]) * u.solMass,
        b=np.array([4, 5, 6]) * u.au,
        v=np.array([7, 8, 9]) * u.km / u.s,
    )
    assert np.all(stars.m == np.array([1, 2, 3]) * u.solMass)
    assert np.all(stars.b == np.array([4, 5, 6]) * u.au)
    assert np.all(stars.v == np.array([7, 8, 9]) * u.km / u.s)


def test_initialize_Stars_with_lists_and_shapes():
    stars = airball.Stars(
        m=[[1, 2, 3], [1, 2, 3]], b=[[4, 5, 6], [4, 5, 6]], v=[[7, 8, 9], [7, 8, 9]]
    )
    assert np.all(stars.m == np.array([[1, 2, 3], [1, 2, 3]]) * u.solMass)
    assert np.all(stars.b == np.array([[4, 5, 6], [4, 5, 6]]) * u.au)
    assert np.all(stars.v == np.array([[7, 8, 9], [7, 8, 9]]) * u.km / u.s)


def test_initialize_Stars_with_lists_and_odd_shapes():
    with pytest.raises(airball.stars.ListLengthException):
        airball.Stars(m=[[1, 2, 3], [1, 2, 3]], b=[[4, 5], [4, 6]], v=[7, 8, 9])


def test_initialize_Stars_with_ndarray_shapes():
    stars = airball.Stars(
        m=np.array([[1, 2, 3], [1, 2, 3]]) * u.solMass,
        b=np.array([[4, 5, 6], [4, 5, 6]]) * u.au,
        v=np.array([[7, 8, 9], [7, 8, 9]]) * u.km / u.s,
    )
    assert np.all(stars.m == np.array([[1, 2, 3], [1, 2, 3]]) * u.solMass)
    assert np.all(stars.b == np.array([[4, 5, 6], [4, 5, 6]]) * u.au)
    assert np.all(stars.v == np.array([[7, 8, 9], [7, 8, 9]]) * u.km / u.s)


def test_initialize_Stars_with_ndarray_odd_shapes():
    with pytest.raises(airball.stars.ListLengthException):
        airball.Stars(
            m=np.array([[1, 2, 3], [1, 2, 3]]) * u.solMass,
            b=np.array([[4, 5], [4, 6]]) * u.au,
            v=np.array([7, 8, 9]) * u.km / u.s,
        )
    with pytest.raises(airball.stars.ListLengthException):
        airball.Stars(
            m=np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]) * u.solMass,
            b=np.array([[4, 5], [4, 6]]) * u.au,
            v=np.array([7, 8]) * u.km / u.s,
        )
    with pytest.raises(airball.stars.ListLengthException):
        airball.Stars(
            m=np.array([[1, 2, 3], [1, 2, 3]]) * u.solMass,
            b=np.array([[4, 5], [4, 6]]) * u.au,
            v=np.array([7, 8]) * u.km / u.s,
        )


def test_initialize_Stars_with_mixed_lists():
    stars = airball.Stars(m=[1, 2, 3], b=np.array([4, 5, 6]), v=[7, 8, 9] * u.km / u.s)
    assert np.all(stars.m == np.array([1, 2, 3]) * u.solMass)
    assert np.all(stars.b == np.array([4, 5, 6]) * u.au)
    assert np.all(stars.v == np.array([7, 8, 9]) * u.km / u.s)


def test_initialize_Stars_with_mixed_lists_and_floats():
    stars = airball.Stars(m=np.array([4, 5, 6]), b=1, v=7 * u.km / u.s)
    assert np.all(stars.m == np.array([4, 5, 6]) * u.solMass)
    assert np.all(stars.b == np.array([1, 1, 1]) * u.au)
    assert np.all(stars.v == np.array([7, 7, 7]) * u.km / u.s)

    stars = airball.Stars(m=1 * u.solMass, b=np.array([4, 5, 6]), v=7)
    assert np.all(stars.m == np.array([1, 1, 1]) * u.solMass)
    assert np.all(stars.b == np.array([4, 5, 6]) * u.au)
    assert np.all(stars.v == np.array([7, 7, 7]) * u.km / u.s)


def test_initialize_Stars_with_floats_and_size():
    stars = airball.Stars(m=1, b=3.0, v=7 * u.km / u.s, size=3)
    assert np.all(stars.m == np.array([1, 1, 1]) * u.solMass)
    assert np.all(stars.b == np.array([3, 3, 3]) * u.au)
    assert np.all(stars.v == np.array([7, 7, 7]) * u.km / u.s)


def test_initialize_Stars_with_uneven_lists():
    with pytest.raises(airball.stars.ListLengthException):
        airball.Stars(m=[1], b=[4, 5, 6], v=[7, 8, 9])
    with pytest.raises(airball.stars.ListLengthException):
        airball.Stars(m=[1, 2, 3], b=[4], v=[7, 8, 9])
    with pytest.raises(airball.stars.ListLengthException):
        airball.Stars(m=[1, 2, 3], b=[4, 5, 6], v=[7])


def test_initialize_Stars_with_unspecified_parameters():
    with pytest.raises(airball.stars.UnspecifiedParameterException):
        airball.Stars(b=[4, 5, 6], v=[7, 8, 9])
    with pytest.raises(airball.stars.UnspecifiedParameterException):
        airball.Stars(m=[1, 2, 3], v=[7, 8, 9])
    with pytest.raises(airball.stars.UnspecifiedParameterException):
        airball.Stars(m=[1, 2, 3], b=[4, 5, 6])


def test_initialize_Stars_with_floats_and_unspecified_size():
    airball.Stars(m=1, b=3, v=7)


def test_initialize_Stars_with_lists_and_specified_size():
    with pytest.raises(airball.stars.ListLengthException):
        airball.Stars(m=[1, 2], v=[7, 8, 9], size=4)


def test_initialize_Stars_with_custom_object():
    # Test using a custom object that defines __len__.
    class CustomObject:
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        @property
        def shape(self):
            return (len(self.data),)

    # Create an instance of the custom object
    my_object = CustomObject([1, 2, 3])
    with pytest.raises(airball.stars.IncompatibleListException):
        airball.Stars(m=my_object, b=[4, 5, 6], v=[7, 8, 9])
        airball.Stars(m=[1, 2, 3], b=my_object, v=[7, 8, 9])
        airball.Stars(m=[1, 2, 3], b=[4, 5, 6], v=my_object)


def test_initialize_Stars_with_custom_object_no_shape():
    # Test using a custom object that defines __len__.
    class CustomObject:
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

    # Create an instance of the custom object
    my_object = CustomObject([1, 2, 3])
    with pytest.raises(airball.stars.IncompatibleListException):
        airball.Stars(m=my_object, b=[4, 5, 6], v=[7, 8, 9])
        airball.Stars(m=[1, 2, 3], b=my_object, v=[7, 8, 9])
        airball.Stars(m=[1, 2, 3], b=[4, 5, 6], v=my_object)


def test_initialize_Stars_with_custom_object_different_list_length():
    # Test using a custom object that defines __len__.
    class CustomObject:
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        @property
        def shape(self):
            return (len(self.data),)

    # Create an instance of the custom object
    my_object = CustomObject([1, 2])
    with pytest.raises(airball.stars.ListLengthException):
        airball.Stars(m=my_object, b=[4, 5, 6], v=[7, 8, 9])
        airball.Stars(m=[1, 2, 3], b=my_object, v=[7, 8, 9])
        airball.Stars(m=[1, 2, 3], b=[4, 5, 6], v=my_object)


def test_initialize_Stars_with_custom_object_no_shape_different_list_length():
    # Test using a custom object that defines __len__.
    class CustomObject:
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

    # Create an instance of the custom object
    my_object = CustomObject([1, 2])
    with pytest.raises(airball.stars.ListLengthException):
        airball.Stars(m=my_object, b=[4, 5, 6], v=[7, 8, 9])
        airball.Stars(m=[1, 2, 3], b=my_object, v=[7, 8, 9])
        airball.Stars(m=[1, 2, 3], b=[4, 5, 6], v=my_object)


def test_initialize_Stars_with_large_ndarrays():
    ones = np.ones(int(1e7))
    stars = airball.Stars(m=ones, b=2 * ones, v=3 * ones)
    assert np.all(stars.m == ones * u.solMass)
    assert np.all(stars.b == 2 * ones * u.au)
    assert np.all(stars.v == 3 * ones * u.km / u.s)


################################################
################################################
##########  INDEXING AND SLICING TESTS  ########
################################################
################################################


def test_access_Stars_using_ints():
    stars = airball.Stars(
        m=[1, 2, 3],
        b=[4, 5, 6],
        v=[7, 8, 9],
        inc=[3, 2, 1],
        omega=[6, 5, 4],
        Omega=[9, 8, 7],
    )
    assert stars[0] == airball.Star(m=1, b=4, v=7, inc=3, omega=6, Omega=9)


def test_access_multidimensional_Stars_using_ints():
    stars = airball.Stars(
        m=[[1, 2, 3], [2, 2, 3]],
        b=[[4, 5, 6], [5, 5, 6]],
        v=[[7, 8, 9], [8, 8, 9]],
        inc=[[3, 2, 1], [4, 2, 1]],
        omega=[[6, 5, 4], [7, 5, 4]],
        Omega=[[9, 8, 7], [10, 8, 7]],
    )
    assert stars[0] == airball.Stars(
        m=[1, 2, 3],
        b=[4, 5, 6],
        v=[7, 8, 9],
        inc=[3, 2, 1],
        omega=[6, 5, 4],
        Omega=[9, 8, 7],
    )


################################################
################################################
##########  EQUALITY TESTS  ####################
################################################
################################################


def test_Star_equality():
    star = airball.Star(m=1, b=4, v=7, inc=3, omega=6, Omega=9)
    assert star == airball.Star(m=1, b=4, v=7, inc=3, omega=6, Omega=9)
    assert star != airball.Star(m=1, b=4, v=7, inc=3, omega=6, Omega=0)
    assert hash(star) == hash(airball.Star(m=1, b=4, v=7, inc=3, omega=6, Omega=9))
    assert hash(star) != hash(airball.Star(m=1, b=4, v=7, inc=3, omega=6, Omega=0))


def test_Stars_equality():
    stars1 = airball.Stars(
        m=[1, 2, 3],
        b=[4, 5, 6],
        v=[7, 8, 9],
        inc=[3, 2, 1],
        omega=[6, 5, 4],
        Omega=[9, 8, 7],
    )
    stars2 = airball.Stars(
        m=[1, 2, 3],
        b=[4, 5, 6],
        v=[7, 8, 9],
        inc=[3, 2, 1],
        omega=[6, 5, 4],
        Omega=[9, 8, 7],
    )
    assert stars1 == stars2
    assert stars1[:2] == stars2[:2]
    assert hash(stars1) == hash(stars2)
    stars2.sort("m")
    assert stars1 == stars2
    stars2.sort("inc")
    assert stars1 != stars2
    assert hash(stars1) != hash(stars2)


def test_Stars_from_StellarEnvironment_equality():
    oc = airball.OpenCluster()
    stars1 = oc.random_star(size=300)
    stars2 = stars1.copy()
    assert stars1 == stars2
    assert stars1[:2] == stars2[:2]
    assert hash(stars1) == hash(stars2)
    stars2.sort("m")
    assert stars1 != stars2
    assert hash(stars1) != hash(stars2)
