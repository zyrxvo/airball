import numpy as _numpy
from scipy.integrate import quad as _quad
from . import units as u
from . import tools


def chabrier_2003_single(x, A=0.158):
    """
    Chabrier 2003 IMF for single stars.
    This function calculates the probability density for a given mass value (x) based on the Chabrier 2003 IMF equation.

    Parameters:
    - x: Mass value (float).

    Returns:
    - Probability density at the given mass value.
    """
    return (A / x) * _numpy.exp(-((_numpy.log10(x) - _numpy.log10(0.079)) ** 2) / (2 * 0.69 ** 2))

def salpeter_1955(x, A):
    """
    Salpeter 1955 IMF for single stars.
    This function calculates the probability density for a given mass value (x) based on the Salpeter 1955 IMF equation.

    Parameters:
    - x: Mass value (float).

    Returns:
    - Probability density at the given mass value.
    """
    return A * x ** -2.3


class IMF():
  """
  Class representing an Initial Mass Function (IMF).
  It generates random masses based on a given mass function and provides various properties and methods for manipulating and analyzing the IMF.

  Parameters:
  - min_mass: Minimum mass value of the IMF range (float or astropy.units.Quantity).
  - max_mass: Maximum mass value of the IMF range (float or astropy.units.Quantity).
  - mass_function: Mass function to use for the IMF (optional).
  - unit: Unit of mass (optional, defaults to solar masses).
  - number_samples: Number of samples to use for interpolating the CDF (default: 100).
  - seed: Value to seed the random number generator with (optional, int, or None to turn off).
  """

  def __init__(self, min_mass, max_mass, mass_function=None, unit=u.solMass, number_samples=100, seed=None):
    self._number_samples = int(number_samples)
    self._seed = seed
    self.unit = unit if tools.isUnit(unit) else u.solMass

    # Convert min_mass and max_mass to specified unit if they are Quantity objects, otherwise assume they are already in the correct unit
    self._min_mass = min_mass.to(self.unit) if tools.isQuantity(min_mass) else min_mass * self.unit
    if self._min_mass.value <= 0: raise Exception('Cannot have minimum mass value be less than or equal to 0.')
    self._max_mass = max_mass.to(self.unit) if tools.isQuantity(max_mass) else max_mass * self.unit

    # Determine the probability distribution function (PDF) based on the given mass function or default to a piecewise Chabrier 2003 and Salpeter 1955
    if mass_function is None: mass_function = lambda x: _numpy.where(x < 1, chabrier_2003_single(x), salpeter_1955(x, chabrier_2003_single(1)))
    self._imf = mass_function

    # Recalculate the IMF properties based on the updated parameters
    self._recalculate()

  def _recalculate(self):
    """
    Recalculates the IMF properties based on the current parameters.
    This function updates the normalization factor, normalized PDF, cumulative distribution function (CDF),
    mass values, and IMF values based on the current min_mass, max_mass, and PDF.
    """
    # Calculate the normalization factor for the PDF
    pdf = _numpy.vectorize(lambda x: _numpy.where(x < self.min_mass.value, 0, _numpy.where(x > self.max_mass.value, 0, self._imf(x))))
    normalization_factor = _quad(pdf, self._min_mass.value, self._max_mass.value)[0]

    # Create a normalized PDF
     
    npdf = lambda x: pdf(x) / normalization_factor
    self._npdf = _numpy.vectorize(npdf)

    # Create a cumulative distribution function (CDF)
    cdf = lambda x: _numpy.where(x < self.min_mass.value, 0, _numpy.where(x > self.max_mass.value, 1,  _quad(self._npdf, self._min_mass.value, x)[0]))
    self._cdf = _numpy.vectorize(cdf)

    # Generate logarithmically spaced mass values between min_mass and max_mass
    self._masses = _numpy.logspace(_numpy.log10(self._min_mass.value), _numpy.log10(self._max_mass.value), self.number_samples)

    # Calculate the CDF and IMF values for the mass values
    self._CDF = self._cdf(self._masses)
    self._norm_imf = self._npdf(self._masses)

  def random_mass(self, size=1, **kwargs):
    """
    Generates random mass values from the IMF.

    Parameters:
    - size: Number of random mass values to generate (default: 1).

    Returns:
    - Randomly generated mass value(s) from the IMF.
    """
    if 'seed' in kwargs: self.seed = kwargs['seed']
    if self.seed != None: _numpy.random.seed(self.seed)
    rand_masses = _numpy.interp(_numpy.random.uniform(size=size), self._CDF, self._masses) * self.unit
    if isinstance(size, tuple): return rand_masses
    elif size > 1: return rand_masses
    else: return rand_masses[0]

  def masses(self, number_samples, endpoint=True):
    """
    Generates an array of mass values logarithmically spanning the IMF range.

    Parameters:
    - number_samples: Number of mass values to generate.

    Returns:
    - numpy array of mass values logarithmically spanning the IMF range.
    """
    return _numpy.logspace(_numpy.log10(self._min_mass.value), _numpy.log10(self._max_mass.value), number_samples, endpoint=endpoint)

  @property
  def min_mass(self):
    """
    Minimum mass value of the IMF range.
    """
    return self._min_mass

  @min_mass.setter
  def min_mass(self, value):
    """
    Setter for the minimum mass value of the IMF range.
    Recalculates the IMF properties when the min_mass value is updated.

    Parameters:
    - value: New minimum mass value (float or astropy.units.Quantity).
    """
    value = value.to(self.unit) if tools.isQuantity(value) else value * self.unit
    if value.value <= 0: raise Exception('Cannot have minimum mass value be less than or equal to 0.')
    self._min_mass = value
    self._recalculate()

  @property
  def max_mass(self):
    """
    Maximum mass value of the IMF range.
    """
    return self._max_mass

  @max_mass.setter
  def max_mass(self, value):
    """
    Setter for the maximum mass value of the IMF range.
    Recalculates the IMF properties when the max_mass value is updated.

    Parameters:
    - value: New maximum mass value (float or astropy.units.Quantity).
    """
    self._max_mass = value.to(self.unit) if tools.isQuantity(value) else value * self.unit
    self._recalculate()

  @property
  def median_mass(self):
    """
    Median mass value of the IMF.
    """
    return _numpy.interp(0.5, self._CDF, self._masses) * self.unit
  
  @property
  def seed(self):
    """
    Seed for the random number generator.
    """
    return self._seed

  @seed.setter
  def seed(self, value):
    """
    Setter for the seed for the random number generator 

    Parameters:
    - value: New seed for the random number generator (int, or None to turn off)
    """
    self._seed = value

  @property
  def number_samples(self):
    """
    Seed for the random number generator.
    """
    return self._number_samples

  @number_samples.setter
  def number_samples(self, value):
    """
    Setter for the seed for the random number generator 

    Parameters:
    - value: New seed for the random number generator (int, or None to turn off)
    """
    self._number_samples = int(value)
    self._recalculate()

  @property
  def cdf(self):
    """
    Cumulative distribution function (CDF) of the IMF.
    """
    return lambda x: _numpy.interp(x, self._masses, self._CDF)

  @property
  def CDF(self):
    """
    Cumulative distribution function (CDF) of the IMF.
    """
    return self.cdf

  @property
  def pdf(self):
    """
    Probability density function (PDF) of the IMF.
    """
    return self._npdf

  @property
  def PDF(self):
    """
    Probability density function (PDF) of the IMF.
    """
    return self.pdf

  @property
  def imf(self):
    """
    Probability density function (PDF) of the IMF.
    """
    return self._imf

  @property
  def IMF(self):
    """
    Probability density function (PDF) of the IMF.
    """
    return self.imf
