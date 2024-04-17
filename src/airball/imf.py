import numpy as _np
import types as _types
from scipy.integrate import quad as _quad
from . import units as _u
from . import tools as _tools

class Distribution():
  '''
  Base class for defining a probability distribution function.
  This class is used to wrap a probability distribution function and its parameters into a single object.

  Args:
    mass_function (function): Probability distribution function.
    args (list): List of arguments for the probability distribution function.

  Returns:
    pdf (float or ndarray): Probability density at the given mass value(s).
  
  Example:
    ```python
    import airball
    mf = airball.imf.Distribution(lambda x, A: A * x, [1])
    imf = airball.IMF(0.1, 100, mass_function=mf)
    imf.random_mass()
    ```

  '''

  def __init__(self, mass_function, args):
    self.mass_function = mass_function
    self.params = args
  
  def __call__(self, x):
    return self.mass_function(x, *self.params)

class chabrier_2003_single(Distribution):
  """
  Chabrier 2003 IMF for single stars.
  This function calculates the probability density for a given mass value (x) based on Equation (17) of Chabrier 2003 for an IMF for single stars.

  $$PDF(x) = \\frac{A}{x} \\exp\\left[-\\frac{(\\log_{10}(x) - \\log_{10}(0.079))^2 }{ 2 \\cdot 0.69^2}\\right]$$

  Args:
    A (float, optional): Normalization factor. Default is 0.158.

  Returns:
    pdf (float or ndarray): Probability density at the given mass value(s).

  Example:
    ```python
    import airball
    imf = airball.IMF(0.1, 100, mass_function=airball.imf.chabrier_2003_single(A=0.158))
    imf.random_mass()
    ```
  """
  def __init__(self, A=0.158):
    super().__init__(self._chabrier_2003_single, [A])
  
  def _chabrier_2003_single(self, x, A=0.158):
    return (A / x) * _np.exp(-((_np.log10(x) - _np.log10(0.079)) ** 2) / (2 * 0.69 ** 2))

class salpeter_1955(Distribution):
  """
  Salpeter 1955 IMF for single stars.
  This function calculates the probability density for a given mass value (x) based on the Salpeter 1955 IMF equation.

  $$PDF(x) = A x^{-2.3}$$

  Args:
    A (float): Normalization factor.

  Returns:
    pdf (float or ndarray): Probability density at the given mass value(s).
  
  Example:
    ```python
    import airball
    imf = airball.IMF(0.1, 100, mass_function=airball.imf.salpeter_1955(A=1))
    imf.random_mass()
    ```
  """
  def __init__(self, A):
    super().__init__(self._salpeter_1955, [A])
  
  def _salpeter_1955(self, x, A):
    return A * x ** -2.3

class default_mass_function(Distribution):
  '''
  Default mass function for an IMF. This function is a piecewise function that uses the IMF for single stars from Chabrier 2003 for masses less than 1 solar mass and the Salpeter 1955 IMF for masses greater than 1 solar mass.

  Args:
    x (float or ndarray): Mass value(s).

  Returns:
    pdf (float or ndarray): Probability density at the given mass value(s).

  Example:
    ```python
    import airball
    imf = airball.IMF(0.1, 100, mass_function=airball.imf.default_mass_function())
    imf.random_mass()
    ```
  '''
  def __init__(self):
    super().__init__(self._default_mass_function, [])
  
  def _default_mass_function(self, x):
    chabrier03 = chabrier_2003_single(A=0.158)
    salpeter55 = salpeter_1955(A=chabrier03(1))
    return _np.where(x < 1, chabrier03(x), salpeter55(x))

class uniform(Distribution):
  '''
  Uniform IMF.
  This function calculates the probability density for a given mass value (x) based on a uniform IMF.
  
  $$PDF(x) = 1$$
  '''
  def __init__(self):
    super().__init__(self._uniform)
  
  def _uniform(self, x):
    return x * 0 + 1

class power_law(Distribution):
  '''
  Power law IMF.
  This function calculates the probability density for a given mass value (x) based on a power law IMF.
  
  $$PDF(x) = A x^\\alpha$$
  
  Args:
    alpha (float): Power law index.
    A (float): Normalization factor.

  Returns:
    pdf (float or ndarray): Probability density at the given mass value(s).
  '''
  def __init__(self, alpha, A):
    super().__init__(self._power_law, [alpha, A])
  
  def _power_law(self, x, alpha, A):
    return A * x ** alpha

class broken_power_law(Distribution):
  '''
  Broken power law IMF.
  This function calculates the probability density for a given mass value (x) based on a broken power law IMF.
  
  $$PDF(x) = \\begin{cases} A x^\\alpha & x < x_0 \\\\ A x_0^{\\beta - \\alpha} x^\\beta & x \\geq x_0 \\end{cases}$$
  
  Args:
    alpha (float): Power law index for $x < x_0$.
    beta (float): Power law index for $x ≥ x_0$.
    A (float): Normalization factor.
    x_0 (float): Break point between the two power laws.
    
  Returns:
    pdf (float or ndarray): Probability density at the given mass value(s).
  '''
  def __init__(self, alpha, beta, A, x_0):
    super().__init__(self._broken_power_law, [alpha, beta, A, x_0])

  def _broken_power_law(self, x, alpha, beta, A, x_0):
    return _np.where(x < x_0, A * x ** alpha, A * x_0 ** (beta - alpha) * x ** beta)

class lognormal(Distribution):
  '''
  Lognormal IMF.
  This function calculates the probability density for a given mass value (x) based on a lognormal IMF.
  
  $$PDF(x) = A \\exp\\left[-\\frac{(\\log_{10}(x) - \\mu)^2 }{ 2 \\sigma^2}\\right]$$
  
  Args:
    mu (float): Mean of the lognormal distribution.
    sigma (float): Standard deviation of the lognormal distribution.
    A (float): Normalization factor.
    
  Returns:
    pdf (float or ndarray): Probability density at the given mass value(s).
  '''
  def __init__(self, mu, sigma, A):
    super().__init__(self._lognormal, [mu, sigma, A])
  
  def _lognormal(self, x, mu, sigma, A):
    return A * _np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

class loguniform(Distribution):
  '''
  Loguniform IMF.
  This function calculates the probability density for a given mass value (x) based on a loguniform IMF.
  
  $$PDF(x) = \\frac{A}{x}$$
  
  Args:
    A (float): Normalization factor.
    
  Returns:
    pdf (float or ndarray): Probability density at the given mass value(s).
  '''
  def __init__(self, A):
    super().__init__(self._loguniform, [A])

  def _loguniform(self, x, A):
    return A / x

class IMF():
  """
  Class representing an Initial Mass Function (IMF).
  It generates random masses based on a given mass function and provides various properties and methods for manipulating and analyzing the IMF.

  Args:
    min_mass (float): Minimum mass value of the IMF range.
    max_mass (float): Maximum mass value of the IMF range.
    mass_function (function, optional): Mass function to use for the IMF. Default is a piecewise Chabrier 2003 and Salpeter 1955.
    unit (Unit, optional): Unit of mass. Default is solar masses.
    number_samples (float, optional): Number of samples to use for interpolating the CDF. Default is 100.
    seed (float, optional): Value to seed the random number generator with. Default is None.

  Attributes:
    min_mass (float): Minimum mass value of the IMF range.
    max_mass (float): Maximum mass value of the IMF range.
    median_mass (float): Median mass value of the IMF.
    seed (float): Value to seed the random number generator with.
    number_samples (float): Number of samples to use for interpolating the CDF.
    unit (Unit): Unit of mass.
    normalization_factor (float): Normalization factor for the PDF.
    masses (ndarray): Mass values logarithmically spanning the IMF range.
    CDF (function): Cumulative distribution function (CDF) of the IMF.
    PDF (function): Normalized probability density function (PDF) of the IMF.
    IMF (function): Initial mass function (IMF) of the IMF.
  """

  def __init__(self, min_mass, max_mass, mass_function=None, unit=_u.solMass, number_samples=100, seed=None):
    self._number_samples = int(number_samples)
    self._seed = seed
    self.unit = unit if _u.isUnit(unit) else _u.solMass

    # Convert min_mass and max_mass to specified unit if they are Quantity objects, otherwise assume they are already in the correct unit
    self._min_mass = min_mass.to(self.unit) if _tools.isQuantity(min_mass) else min_mass * self.unit
    if self._min_mass.value <= 0: raise ValueError('Cannot have minimum mass value be less than or equal to 0.')
    self._max_mass = max_mass.to(self.unit) if _tools.isQuantity(max_mass) else max_mass * self.unit
    if self._max_mass <= self._min_mass: raise ValueError('Cannot have maximum mass value be less than or equal to minimum mass.')
    if self._max_mass.value <= 0: raise ValueError('Cannot have maximum mass value be less than or equal to 0.')

    # Determine the probability distribution function (PDF) based on the given mass function or default to a piecewise Chabrier 2003 and Salpeter 1955
    if mass_function is None: mass_function = default_mass_function()
    elif not isinstance(mass_function, (_types.FunctionType, Distribution)): raise ValueError('mass_function must be a function or a Distribution object.')
    self.initial_mass_function = mass_function

    # Recalculate the IMF properties based on the updated parameters
    self._recalculate()

  ### Define the PDF, normalized PDF, and CDF functions for the IMF. These functions are vectorized so they can accept arrays of values. ###
  def _pdf_(self, x): return _np.where(x < self.min_mass.value, 0, _np.where(x > self.max_mass.value, 0, self.imf(x)))
  def _probability_density_function(self, x): return _np.vectorize(self._pdf_)(x)
  def _npdf_(self, x): return self._pdf_(x) / self.normalization_factor
  def _normalized_probability_density_function(self, x): return _np.vectorize(self._npdf_)(x)
  def _cdf_(self, x): return _np.where(x < self.min_mass.value, 0, _np.where(x > self.max_mass.value, 1,  _quad(self._normalized_probability_density_function, self.min_mass.value, x)[0]))
  def _cumulative_distribution_function(self, x): return _np.vectorize(self._cdf_)(x)

  def _recalculate(self):
    """
    Recalculates the IMF properties based on the current parameters.
    This function updates the normalization factor, normalized PDF, cumulative distribution function (CDF),
    mass values, and IMF values based on the current min_mass, max_mass, and PDF.
    """
    # Calculate the normalization factor for the PDF
    self.normalization_factor = _quad(self._probability_density_function, self._min_mass.value, self._max_mass.value)[0]
    # Generate logarithmically spaced mass values between min_mass and max_mass
    self._masses = _np.logspace(_np.log10(self._min_mass.value), _np.log10(self._max_mass.value), self.number_samples)
    # Calculate the CDF for the mass values
    self._CDF = self._cumulative_distribution_function(self._masses)

  def random_mass(self, size=1, **kwargs):
    """
    Generates random mass values from the IMF.

    Args:
      size (int or tuple): Number of mass values to generate. If size is a tuple, it is interpreted as array dimensions. Default: 1.

    Keyword Args:
      seed (int): Seed for the random number generator. Default: None.

    Returns:
      masses (float or ndarray): Randomly generated mass value(s) from the IMF.

    Example:
      ```python
      import airball
      imf = airball.IMF(0.1, 100)
      imf.random_mass()
      ```
    """
    if 'seed' in kwargs: self.seed = kwargs['seed']
    if self.seed != None: _np.random.seed(self.seed)
    
    if isinstance(size, tuple): size = tuple([int(i) for i in size])
    else: size = int(size)

    rand_masses = _np.interp(_np.random.uniform(size=size), self._CDF, self._masses) * self.unit
    if isinstance(size, tuple): return rand_masses
    elif size > 1: return rand_masses
    else: return rand_masses[0]

  def masses(self, number_samples, endpoint=True):
    """
    Convenience function for generating an array of mass values logarithmically spanning the IMF range.

    Args:
      number_samples (int): Number of mass values to generate.
      endpoint (bool, optional): Whether to include the max_mass value in the array. Default: True.

    Returns:
      masses (ndarray): numpy array of mass values logarithmically spanning the IMF range.
    """
    return _np.logspace(_np.log10(self._min_mass.value), _np.log10(self._max_mass.value), int(number_samples), endpoint=endpoint)

  @property
  def min_mass(self):
    """
    The minimum mass value of the IMF range.
    Recalculates the IMF properties when the `min_mass` value is updated.

    Args:
      value (float): New minimum mass value. Must be greater than 0. Units are assumed to be the same as the IMF unit.
    """
    return self._min_mass

  @min_mass.setter
  def min_mass(self, value):
    value = value.to(self.unit) if _tools.isQuantity(value) else value * self.unit
    if value.value <= 0: raise ValueError('Cannot have minimum mass value be less than or equal to 0.')
    self._min_mass = value
    self._recalculate()

  @property
  def max_mass(self):
    """
    The maximum mass value of the IMF range.
    Recalculates the IMF properties when the `max_mass` value is updated.

    Args:
      value (float): New maximum mass value. Must be greater than minimum mass value. Units are assumed to be the same as the IMF unit.
    """
    return self._max_mass

  @max_mass.setter
  def max_mass(self, value):
    value = value.to(self.unit) if _tools.isQuantity(value) else value * self.unit
    if value.value <= 0: raise ValueError('Cannot have maximum mass value be less than or equal to 0.')
    if value.value <= self.min_mass: raise ValueError('Cannot have maximum mass value be less than or equal to minimum mass value.')
    self._max_mass = value.to(self.unit) if _tools.isQuantity(value) else value * self.unit
    self._recalculate()

  @property
  def median_mass(self):
    """
    Median mass value of the IMF.
    """
    return _np.interp(0.5, self._CDF, self._masses) * self.unit
  
  @property
  def seed(self):
    """
    The seed for the random number generator 

    Args:
      value (int or None): New seed for the random number generator (int, or None to turn off).
    """
    return self._seed

  @seed.setter
  def seed(self, value):
    self._seed = value

  @property
  def number_samples(self):
    """
    The number of samples to use for interpolating the CDF.
    Recalculates the IMF properties when the `number_samples` value is updated.

    Args:
      value (int): New number of samples to use for interpolating the CDF.
    """
    return self._number_samples

  @number_samples.setter
  def number_samples(self, value):
    self._number_samples = int(value)
    self._recalculate()

  def cdf(self, x):
    """Cumulative distribution function (CDF) of the IMF."""
    return _np.interp(x, self._masses, self._CDF)

  @property
  def CDF(self):
    return self.cdf

  def pdf(self, x):
    """Normalized probability density function (PDF) of the IMF."""
    return self._normalized_probability_density_function(x)

  @property
  def PDF(self):
    return self.pdf

  def imf(self, x):
    """The initial mass function (IMF) of the IMF."""
    return self.initial_mass_function(x)

  @property
  def IMF(self):
    return self.imf

  def __eq__(self, other):
    # Overrides the default implementation
    if isinstance(other, IMF):
        return (self.min_mass == other.min_mass and self.max_mass == other.max_mass and self.initial_mass_function == other.initial_mass_function and self.unit == other.unit and self.number_samples == other.number_samples and self.seed==other.seed)
    else:
      return NotImplemented
  
  def __hash__(self):
    # Overrides the default implementation
    data = []
    for d in sorted(self.__dict__.items()):
        try: data.append((d[0], tuple(d[1])))
        except: data.append(d)
    data = tuple(data)
    return hash(data)
  

  def summary(self, returned=False):
    ''' 
    Prints a compact summary of the current stats of the Stellar Environment object.
    '''
    s = f"<{self.__module__}.{type(self).__name__} object at {hex(id(self))}"
    s += f", m= {self.min_mass.value:,.2f}-{self.max_mass.value:,.1f} {self.unit}"
    s += f", IMF= {self.initial_mass_function.__class__.__name__}" if isinstance(self.initial_mass_function, Distribution) else ', IMF= custom'
    s += ">"
    if returned: return s
    else: print(s)
  
  def __str__(self):
    return self.summary(returned=True)
  
  def __repr__(self):
    return self.summary(returned=True)
