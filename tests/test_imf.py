import pytest
import astropy.units as u
import airball.imf as imf
import numpy as np

def test_imf_init():
    # Test initialization with minimum required arguments
    my_imf = imf.IMF(min_mass=0.1, max_mass=100)
    assert my_imf._number_samples == 1000  # default value
    assert my_imf._seed is None  # default value
    assert my_imf.unit == u.solMass  # default value

    # Test initialization with all arguments
    my_imf = imf.IMF(min_mass=0.1, max_mass=100, mass_function=imf.salpeter_1955(A=1), unit=u.solMass, number_samples=20, seed=42)
    assert my_imf._number_samples == 20
    assert my_imf._seed == 42
    assert my_imf.unit == u.solMass

    # Test initialization with invalid arguments
    with pytest.raises(ValueError):
        my_imf = imf.IMF(min_mass=-0.1, max_mass=100)  # negative min_mass
    with pytest.raises(ValueError):
        my_imf = imf.IMF(min_mass=0.1, max_mass=-100)  # negative max_mass
    with pytest.raises(ValueError):
        my_imf = imf.IMF(min_mass=0.1, max_mass=100, number_samples=-100)  # negative number_samples