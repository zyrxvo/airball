import pytest
import astropy.units as u
import airball.imf as imf

def test_imf_default_init():
    """Test initialization with minimum required arguments"""
    my_imf = imf.IMF(min_mass=0.1, max_mass=100)
    assert my_imf._number_samples == 1000  # default value
    assert my_imf._seed is None  # default value
    assert my_imf.unit == u.solMass  # default value

@pytest.mark.parametrize("seed", (None, 100, 1000))
@pytest.mark.parametrize("number_samples", (10, 100, 500))
@pytest.mark.parametrize("unit", (u.solMass, u.jupiterMass))
@pytest.mark.parametrize("mass_function", (imf.salpeter_1955(A=1), imf.default_mass_function()))
@pytest.mark.parametrize("max_mass", (100, 100 * u.solMass))
def test_imf_init(max_mass, mass_function, unit, number_samples, seed):
    """Test initialization with all arguments"""
    mass_function.unit = unit
    my_imf = imf.IMF(
        min_mass=0.1 * unit,
        max_mass=max_mass,
        mass_function=mass_function,
        unit=unit,
        number_samples=number_samples,
        seed=seed,
    )
    assert my_imf.max_mass == max_mass << unit
    assert my_imf._number_samples == number_samples  
    assert my_imf._seed == seed
    assert my_imf.unit == unit


@pytest.mark.parametrize("number_samples", (100, -100))
@pytest.mark.parametrize("max_mass", (100, -100))
@pytest.mark.parametrize("min_mass", (0.1, -0.1))
def test_imf_init_errors(min_mass, max_mass, number_samples):
    """Test initialization with invalid arguments"""
    if (min_mass * max_mass * number_samples) < 0:
        with pytest.raises(ValueError):
            my_imf = imf.IMF(min_mass, max_mass, number_samples)
