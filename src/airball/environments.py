import rebound as _rebound
import numpy as _numpy
from scipy.stats import uniform as _uniform
from scipy.stats import maxwell as _maxwell
from scipy.optimize import fminbound as _fminbound
from scipy.integrate import quad as _quad

try:
    # Required for Python>=3.9
    from collections.abc import MutableMapping
except:
    from collections import MutableMapping

from . import units
from .flybys import *
from .analytic import *
from .tools import *


def _scale(sigma):
  '''
    Converts velocity dispersion (sigma) to scale factor for Maxwell-Boltzmann distributions.
  '''
  return _numpy.sqrt((_numpy.pi*_numpy.square(sigma))/(3.0*_numpy.pi - 8.0))

class UnitSystem():

  def __init__(self, UNIT_SYSTEM=[]) -> None:
    self._units = {'length': units.au, 'time': units.Myr, 'mass': units.solMass, 'angle': units.rad, 'velocity': units.km/units.s, 'object': units.stars, 'density': units.stars/units.pc**3}
    self.UNIT_SYSTEM = UNIT_SYSTEM
    pass

  @property
  def units(self):
    return self._units

  @property
  def UNIT_SYSTEM(self):
    return self._UNIT_SYSTEM

  @UNIT_SYSTEM.setter
  def UNIT_SYSTEM(self, UNIT_SYSTEM):
    if UNIT_SYSTEM != []:
      lengthUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(units.m)]
      self._units['length'] = lengthUnit[0] if lengthUnit != [] else self._units['length']

      timeUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(units.s)]
      self._units['time'] = timeUnit[0] if timeUnit != [] else self._units['time']

      velocityUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(units.km/units.s)]
      if velocityUnit == [] and timeUnit != [] and lengthUnit != []: velocityUnit = [lengthUnit[0]/timeUnit[0]]
      self._units['velocity'] = velocityUnit[0] if velocityUnit != [] else self._units['velocity']

      massUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(units.kg)]
      self._units['mass'] = massUnit[0] if massUnit != [] else self._units['mass']

      angleUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(units.rad)]
      self._units['angle'] = angleUnit[0] if angleUnit != [] else self._units['angle']

      objectUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(units.stars)]
      self._units['object'] = objectUnit[0] if objectUnit != [] else units.stars

      densityUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(units.stars/units.m**3)]
      densityUnit2 = [this for this in UNIT_SYSTEM if this.is_equivalent(1/units.m**3)]
      if densityUnit == [] and densityUnit2 != []: densityUnit = [self._object_unit * densityUnit2[0]]
      elif densityUnit == [] and objectUnit != [] and lengthUnit != []: densityUnit = [self._units['object']/self._units['length']**3]
      self._units['density'] = densityUnit[0] if densityUnit != [] else self._units['density']
    
    self._UNIT_SYSTEM = list(self._units.values())

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

  def __init__(self, min_mass, max_mass, mass_function=None, unit=units.solMass, number_samples=100, seed=None):
    self.number_samples = number_samples
    self._seed = seed
    self.unit = unit if isUnit(unit) else units.solMass

    # Convert min_mass and max_mass to specified unit if they are Quantity objects, otherwise assume they are already in the correct unit
    self._min_mass = min_mass.to(self.unit) if isQuantity(min_mass) else min_mass * self.unit
    self._max_mass = max_mass.to(self.unit) if isQuantity(max_mass) else max_mass * self.unit

    # Determine the probability distribution function (PDF) based on the given mass function or default to Chabrier 2003 or Salpeter 1955
    self._imf = mass_function
    if self._imf is None: 
      self._imf = lambda x: IMF.chabrier_2003_single(x) if x < 1 else IMF.salpeter_1955(x)
    self._pdf = _numpy.vectorize(self._imf)

    # Recalculate the IMF properties based on the updated parameters
    self._recalculate()

  def chabrier_2003_single(x):
    """
    Chabrier 2003 IMF for single stars.
    This function calculates the probability density for a given mass value (x) based on the Chabrier 2003 IMF equation.

    Parameters:
    - x: Mass value (float).

    Returns:
    - Probability density at the given mass value.
    """
    return (0.158 / x) * _numpy.exp(-((_numpy.log10(x) - _numpy.log10(0.079)) ** 2) / (2 * 0.69 ** 2))

  def salpeter_1955(x):
    """
    Salpeter 1955 IMF for single stars.
    This function calculates the probability density for a given mass value (x) based on the Salpeter 1955 IMF equation.

    Parameters:
    - x: Mass value (float).

    Returns:
    - Probability density at the given mass value.
    """
    return IMF.chabrier_2003_single(1) * x ** -2.3

  def _recalculate(self):
    """
    Recalculates the IMF properties based on the current parameters.
    This function updates the normalization factor, normalized PDF, cumulative distribution function (CDF),
    mass values, and IMF values based on the current min_mass, max_mass, and PDF.
    """
    # Calculate the normalization factor for the PDF
    normalization_factor = _quad(self._pdf, self._min_mass.value, self._max_mass.value)[0]

    # Create a normalized PDF
    npdf = lambda x: self._pdf(x) / normalization_factor
    self._npdf = _numpy.vectorize(npdf)

    # Create a cumulative distribution function (CDF)
    cdf = lambda x: _quad(self._npdf, self._min_mass.value, x)[0]
    self._cdf = _numpy.vectorize(cdf)

    # Generate logarithmically spaced mass values between min_mass and max_mass
    self._masses = _numpy.logspace(_numpy.log10(self._min_mass.value), _numpy.log10(self._max_mass.value), self.number_samples)

    # Calculate the CDF and IMF values for the mass values
    self._CDF = self._cdf(self._masses)
    self._norm_imf = self._npdf(self._masses)

  def random_mass(self, size=1, seed=None):
    """
    Generates random mass values from the IMF.

    Parameters:
    - size: Number of random mass values to generate (default: 1).

    Returns:
    - Randomly generated mass value(s) from the IMF.
    """
    if seed != None: self.seed = seed
    if self.seed != None: _numpy.random.seed(self.seed)
    rand_masses = _numpy.interp(_numpy.random.uniform(size=size), self._CDF, self._masses) * self.unit
    if size > 1: return rand_masses
    else: return rand_masses[0]

  def masses(self, number_samples):
    """
    Generates an array of mass values logarithmically spanning the IMF range.

    Parameters:
    - number_samples: Number of mass values to generate.

    Returns:
    - numpy array of mass values logarithmically spanning the IMF range.
    """
    return _numpy.logspace(_numpy.log10(self._min_mass.value), _numpy.log10(self._max_mass.value), number_samples)

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
    self._min_mass = value.to(self.unit) if isQuantity(value) else value * self.unit
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
    self._max_mass = value.to(self.unit) if isQuantity(value) else value * self.unit
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
  def cdf(self):
    """
    Cumulative distribution function (CDF) of the IMF.
    """
    return self._cdf

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


class Star:
  '''
    This is the AIRBALL Star class.
    It encapsulates the relevant parameters for a given star.
    Only the mass is an quantity intrinsic to the object.
    The impact parameter, velocity, inclination, argument of periastron, and longitude of the ascending node quantities are defined with respect to the host star and plane passing through the star.
  '''
  def __init__(self, m, b, v, inc=None, omega=None, Omega=None, UNIT_SYSTEM=[]) -> None:
    '''
    The Mass, m, (Msun), Impact Parameter, b, (AU), Velocity, v, (km/s)
    Inclination, inc, (rad), Argument of the Periastron, omega (rad), and Longitude of the Ascending Node, Omega, (rad)
    Or define an astropy.units system, i.e. UNIT_SYSTEM = [u.pc, u.Myr, u.solMass, u.rad, u.km/u.s].
    '''
    self.units = UnitSystem(UNIT_SYSTEM)

    if inc == 'uniform' or inc == None: inc = 2.0*_numpy.pi * _uniform.rvs() - _numpy.pi
    if omega == 'uniform' or omega == None: omega = 2.0*_numpy.pi * _uniform.rvs() - _numpy.pi
    if Omega == 'uniform' or Omega == None: Omega = 2.0*_numpy.pi * _uniform.rvs() - _numpy.pi

    self.mass = m
    self.impact_parameter = b
    self.velocity = v
    self.inc = inc
    self.argument_periastron = omega
    self.longitude_ascending_node = Omega

  @property
  def UNIT_SYSTEM(self):
    return self.units.UNIT_SYSTEM

  @UNIT_SYSTEM.setter
  def UNIT_SYSTEM(self, UNIT_SYSTEM):
    self.units.UNIT_SYSTEM = UNIT_SYSTEM

  @property
  def m(self):
    return self._mass.to(self.units.units['mass'])

  @m.setter
  def m(self, value):
    self._mass = value.to(self.units.units['mass']) if isQuantity(value) else value * self.units.units['mass']

  @property
  def mass(self):
    return self.m

  @mass.setter
  def mass(self, value):
    self.m = value
  
  @property
  def b(self):
    return self._impact_parameter.to(self.units.units['length'])

  @b.setter
  def b(self, value):
    self._impact_parameter = value.to(self.units.units['length']) if isQuantity(value) else value * self.units.units['length']

  @property
  def impact_parameter(self):
    return self.b

  @impact_parameter.setter
  def impact_parameter(self, value):
    self.b = value

  @property
  def v(self):
    return self._velocity.to(self.units.units['velocity'])

  @v.setter
  def v(self, value):
    self._velocity = value.to(self.units.units['velocity']) if isQuantity(value) else value * self.units.units['velocity']

  @property
  def velocity(self):
    return self.v

  @velocity.setter
  def velocity(self, value):
    self.v = value

  @property
  def inc(self):
    return self._inclination.to(self.units.units['angle'])

  @inc.setter
  def inc(self, value):
    self._inclination = value.to(self.units.units['angle']) if isQuantity(value) else value * self.units.units['angle']

  @property
  def inclination(self):
    return self.inc

  @inc.setter
  def inclination(self, value):
    self.inc = value

  @property
  def omega(self):
    return self._argument_periastron.to(self.units.units['angle'])

  @omega.setter
  def omega(self, value):
    self._argument_periastron = value.to(self.units.units['angle']) if isQuantity(value) else value * self.units.units['angle']

  @property
  def argument_periastron(self):
    return self.omega

  @argument_periastron.setter
  def argument_periastron(self, value):
    self.omega = value

  @property
  def Omega(self):
    return self._longitude_ascending_node.to(self.units.units['angle'])

  @Omega.setter
  def Omega(self, value):
    self._longitude_ascending_node = value.to(self.units.units['angle']) if isQuantity(value) else value * self.units.units['angle']

  @property
  def longitude_ascending_node(self):
    return self.Omega

  @longitude_ascending_node.setter
  def longitude_ascending_node(self, value):
    self.Omega = value

  @property
  def params(self):
    return [self.m, self.b, self.v, self.inc, self.omega, self.Omega]
  
  @property
  def param_values(self):
    return _numpy.array([self.m.value, self.b.value, self.v.value, self.inc.value, self.omega.value, self.Omega.value])

  def stats(self, returned=False):
    ''' 
    Prints a summary of the current stats of the Star.
    '''
    s = f"<{self.__module__}.{type(self).__name__} object at {hex(id(self))}, "
    s += f"m= {self.mass:1.4g}, "
    s += f"b= {self.impact_parameter:1.4g}, "
    s += f"v= {self.velocity:1.4g}, "
    s += f"inc= {self.inc:1.4g}, "
    s += f"omega= {self.omega:1.4g}, "
    s += f"Omega= {self.Omega:1.4g}>"
    if returned: return s
    else: print(s)
  
  def __str__(self):
    return self.stats(returned=True)
  
  def __repr__(self):
    return self.stats(returned=True)
  
  def __len__(self):
    return -1


class Stars(MutableMapping):
  '''
    This class allows the user to access stars like a dictionary using the star's index.
    Allows for negative indices and slicing.
    The implementation uses astropy.Quantity and numpy ndarrays and only generates a airball.Star object when a single Star is requested.
  '''
  def __init__(self, **kwargs) -> None:
    try: 
      self.units = UnitSystem(kwargs['UNIT_SYSTEM'])
      del kwargs['UNIT_SYSTEM']
    except KeyError: self.units = UnitSystem()

    self._Nstars = 0
    for key in kwargs:
      try: 
        len(kwargs[key])
        if isList(kwargs[key]) and len(kwargs[key]) > self.N:
          self._Nstars = len(kwargs[key])
      except: pass
    if 'size' in kwargs and self.N != 0: raise OverspecifiedParametersException('If lists are given then size cannot be specified.')
    elif 'size' in kwargs: self._Nstars = int(kwargs['size'])
    elif self.N == 0: raise UnspecifiedParameterException('If no lists are given then size must be specified.')
    else: pass

    
    try:
      # Check to see if was key is given.
      value = kwargs['m']
      # Check if length matches other key values.
      if len(value) != self.N: raise ListLengthException()
      # Length of value matches other key values, check if value is a list.
      elif isinstance(value, list):
        # Value is a list, try to turn list of Quantities into a ndarray Quantity.
        try: self._m = _numpy.array([v.to(self.units.units['mass']).value for v in value]) * self.units.units['mass']
        # Value was not a list of Quantities, turn list into ndarray and make a Quantity.
        except: self._m = _numpy.array(value) * self.units.units['mass']
      # Value was not a list, check to see if value is an ndarray.
      elif isinstance(value, _numpy.ndarray):
        # Assume ndarray is a Quantity and try to convert ndarray into given units.
        try: self._m = value.to(self.units.units['mass'])
        # ndarray is not a Quantity so turn ndarray into a Quantity.
        except: self._m = value * self.units.units['mass']
      # Value implements __len__, but is not a list or ndarray.
      else: raise IncompatibleListException()
    # This key is necessary and must be specified, raise and Exception.
    except KeyError: raise UnspecifiedParameterException('Mass, m, must be specified.')
    # Value is not a list, so assume it is an int or float and generate an ndarray of the given value.
    except TypeError: 
      value = value.to(self.units.units['mass']) if isQuantity(value) else value * self.units.units['mass']
      self._m = _numpy.ones(self.N) * value
    # Catch any additional Exceptions.
    except Exception as err: raise err


    try:
      # Check to see if was key is given.
      value = kwargs['b']
      # Check if length matches other key values.
      if len(value) != self.N: raise ListLengthException()
      # Length of value matches other key values, check if value is a list.
      elif isinstance(value, list):
        # Value is a list, try to turn list of Quantities into a ndarray Quantity.
        try: self._b = _numpy.array([v.to(self.units.units['length']).value for v in value]) * self.units.units['length']
        # Value was not a list of Quantities, turn list into ndarray and make a Quantity.
        except: self._b = _numpy.array(value) * self.units.units['length']
      # Value was not a list, check to see if value is an ndarray.
      elif isinstance(value, _numpy.ndarray):
        # Assume ndarray is a Quantity and try to convert ndarray into given units.
        try: self._b = value.to(self.units.units['length'])
        # ndarray is not a Quantity so turn ndarray into a Quantity.
        except: self._b = value * self.units.units['length']
      # Value implements __len__, but is not a list or ndarray.
      else: raise IncompatibleListException()
    # This key is necessary and must be specified, raise and Exception.
    except KeyError: raise UnspecifiedParameterException('Impact Parameter, b, must be specified.')
    # Value is not a list, so assume it is an int or float and generate an ndarray of the given value.
    except TypeError: 
      value = value.to(self.units.units['length']) if isQuantity(value) else value * self.units.units['length']
      self._b = _numpy.ones(self.N) * value
    # Catch any additional Exceptions.
    except Exception as err: raise err

    try:
      # Check to see if was key is given.
      value = kwargs['v']
      # Check if length matches other key values.
      if len(value) != self.N: raise ListLengthException()
      # Length of value matches other key values, check if value is a list.
      elif isinstance(value, list):
        # Value is a list, try to turn list of Quantities into a ndarray Quantity.
        try: self._v = _numpy.array([v.to(self.units.units['velocity']).value for v in value]) * self.units.units['velocity']
        # Value was not a list of Quantities, turn list into ndarray and make a Quantity.
        except: self._v = _numpy.array(value) * self.units.units['velocity']
      # Value was not a list, check to see if value is an ndarray.
      elif isinstance(value, _numpy.ndarray):
        # Assume ndarray is a Quantity and try to convert ndarray into given units.
        try: self._v = value.to(self.units.units['velocity'])
        # ndarray is not a Quantity so turn ndarray into a Quantity.
        except: self._v = value * self.units.units['velocity']
      # Value implements __len__, but is not a list or ndarray.
      else: raise IncompatibleListException()
    # This key is necessary and must be specified, raise and Exception.
    except KeyError: raise UnspecifiedParameterException('Velocity, v, must be specified.')
    # Value is not a list, so assume it is an int or float and generate an ndarray of the given value.
    except TypeError: 
      value = value.to(self.units.units['velocity']) if isQuantity(value) else value * self.units.units['velocity']
      self._v = _numpy.ones(self.N) * value
    # Catch any additional Exceptions.
    except Exception as err: raise err

    try:
      # Check to see if was key is given.
      value = kwargs['inc']
        # Check to see if value for key is string.
      if isinstance(value, str):
        # Value is a string, check to see if value for key is valid.
        if value != 'uniform': raise InvalidValueForKeyException()
        # Value 'uniform' for key is valid, now generate an array of values for key.
        self._inc = (2.0*_numpy.pi * _uniform.rvs(size=self.N) - _numpy.pi) * self.units.units['angle']
      # Value is not a string, check if length matches other key values.
      elif len(value) != self.N: raise ListLengthException()
      # Length of value matches other key values, check if value is a list.
      elif isinstance(value, list):
        # Value is a list, try to turn list of Quantities into a ndarray Quantity.
        try: self._inc = _numpy.array([v.to(self.units.units['angle']).value for v in value]) * self.units.units['angle']
        # Value was not a list of Quantities, turn list into ndarray and make a Quantity.
        except: self._inc = _numpy.array(value) * self.units.units['angle']
      # Value was not a list, check to see if value is an ndarray.
      elif isinstance(value, _numpy.ndarray):
        # Assume ndarray is a Quantity and try to convert ndarray into given units.
        try: self._inc = value.to(self.units.units['angle'])
        # ndarray is not a Quantity so turn ndarray into a Quantity.
        except: self._inc = value * self.units.units['angle']
      # Value implements __len__, but is not a list or ndarray.
      else: raise IncompatibleListException()
    # Key does not exist, assume the user wants an array of values to automatically be generated.
    except KeyError: self._inc = (2.0*_numpy.pi * _uniform.rvs(size=self.N) - _numpy.pi) * self.units.units['angle']
    # Value is not a list, so assume it is an int or float and generate an ndarray of the given value.
    except TypeError: 
      value = value.to(self.units.units['angle']) if isQuantity(value) else value * self.units.units['angle']
      self._inc = _numpy.ones(self.N) * value
    # Catch any additional Exceptions.
    except Exception as err: raise err

    try:
      # Check to see if was key is given.
      value = kwargs['omega']
        # Check to see if value for key is string.
      if isinstance(value, str):
        # Value is a string, check to see if value for key is valid.
        if value != 'uniform': raise InvalidValueForKeyException()
        # Value 'uniform' for key is valid, now generate an array of values for key.
        self._omega = (2.0*_numpy.pi * _uniform.rvs(size=self.N) - _numpy.pi) * self.units.units['angle']
      # Value is not a string, check if length matches other key values.
      elif len(value) != self.N: raise ListLengthException()
      # Length of value matches other key values, check if value is a list.
      elif isinstance(value, list):
        # Value is a list, try to turn list of Quantities into a ndarray Quantity.
        try: self._omega = _numpy.array([v.to(self.units.units['angle']).value for v in value]) * self.units.units['angle']
        # Value was not a list of Quantities, turn list into ndarray and make a Quantity.
        except: self._omega = _numpy.array(value) * self.units.units['angle']
      # Value was not a list, check to see if value is an ndarray.
      elif isinstance(value, _numpy.ndarray):
        # Assume ndarray is a Quantity and try to convert ndarray into given units.
        try: self._omega = value.to(self.units.units['angle'])
        # ndarray is not a Quantity so turn ndarray into a Quantity.
        except: self._omega = value * self.units.units['angle']
      # Value implements __len__, but is not a list or ndarray.
      else: raise IncompatibleListException()
    # Key does not exist, assume the user wants an array of values to automatically be generated.
    except KeyError: self._omega = (2.0*_numpy.pi * _uniform.rvs(size=self.N) - _numpy.pi) * self.units.units['angle']
    # Value is not a list, so assume it is an int or float and generate an ndarray of the given value.
    except TypeError: 
      value = value.to(self.units.units['angle']) if isQuantity(value) else value * self.units.units['angle']
      self._omega = _numpy.ones(self.N) * value
    # Catch any additional Exceptions.
    except Exception as err: raise err

    try:
      # Check to see if was key is given.
      value = kwargs['Omega']
        # Check to see if value for key is string.
      if isinstance(value, str):
        # Value is a string, check to see if value for key is valid.
        if value != 'uniform': raise InvalidValueForKeyException()
        # Value 'uniform' for key is valid, now generate an array of values for key.
        self._Omega = (2.0*_numpy.pi * _uniform.rvs(size=self.N) - _numpy.pi) * self.units.units['angle']
      # Value is not a string, check if length matches other key values.
      elif len(value) != self.N: raise ListLengthException()
      # Length of value matches other key values, check if value is a list.
      elif isinstance(value, list):
        # Value is a list, try to turn list of Quantities into a ndarray Quantity.
        try: self._Omega = _numpy.array([v.to(self.units.units['angle']).value for v in value]) * self.units.units['angle']
        # Value was not a list of Quantities, turn list into ndarray and make a Quantity.
        except: self._Omega = _numpy.array(value) * self.units.units['angle']
      # Value was not a list, check to see if value is an ndarray.
      elif isinstance(value, _numpy.ndarray):
        # Assume ndarray is a Quantity and try to convert ndarray into given units.
        try: self._Omega = value.to(self.units.units['angle'])
        # ndarray is not a Quantity so turn ndarray into a Quantity.
        except: self._Omega = value * self.units.units['angle']
      # Value implements __len__, but is not a list or ndarray.
      else: raise IncompatibleListException()
    # Key does not exist, assume the user wants an array of values to automatically be generated.
    except KeyError: self._Omega = (2.0*_numpy.pi * _uniform.rvs(size=self.N) - _numpy.pi) * self.units.units['angle']
    # Value is not a list, so assume it is an int or float and generate an ndarray of the given value.
    except TypeError: 
      value = value.to(self.units.units['angle']) if isQuantity(value) else value * self.units.units['angle']
      self._Omega = _numpy.ones(self.N) * value
    # Catch any additional Exceptions.
    except Exception as err: raise err
  
  @property
  def N(self):
    return self._Nstars
  
  @property
  def median_mass(self):
    return _numpy.median([mass.value for mass in self.m]) * (self._stars[0].m).unit
  
  @property
  def mean_mass(self):
    return _numpy.mean([mass.value for mass in self.m]) * (self._stars[0].m).unit
  
  def __getitem__(self, key):
    int_types = int, _numpy.integer
    
    if isinstance(key, slice):
        if key == slice(None, None, None): return self
        else: return Stars(m=self.m[key], b=self.b[key], v=self.v[key], inc=self.inc[key], omega=self.omega[key], Omega=self.Omega[key], UNIT_SYSTEM=self.units.UNIT_SYSTEM)

    if isinstance(key, int_types):
        # accept negative indices
        if key < 0: key += self.N
        if key < 0 or key >= self.N: raise AttributeError(f'Index {key} used to access stars out of range.')
        i = int(key)
        return Star(m=self.m[i], b=self.b[i], v=self.v[i], inc=self.inc[i], omega=self.omega[i], Omega=self.Omega[i], UNIT_SYSTEM=self.units.UNIT_SYSTEM)
    
    raise StarInvalidKeyException() 

  def __setitem__(self, key, value):
    star_type = Star, Stars
    if isinstance(value, star_type):
      self.m[key], self.b[key], self.v[key], self.inc[key], self.omega[key], self.Omega[key] = value.params
    else: raise InvalidStarException()
  
  def __delitem__(self, key):
    raise ValueError('Cannot delete Star elements from Stars array.')

  def __iter__(self):
    for i in range(self.N):
      yield Star(m=self.m[i], b=self.b[i], v=self.v[i], inc=self.inc[i], omega=self.omega[i], Omega=self.Omega[i], UNIT_SYSTEM=self.units.UNIT_SYSTEM)

  def __len__(self):
    return self.N
  
  def sortby(self, key):
    inds = _numpy.arange(self.N)
    if key == 'm': inds = _numpy.argsort(self.m)
    elif key == 'b': inds = _numpy.argsort(self.b)
    elif key == 'v': inds = _numpy.argsort(self.v)
    elif key == 'inc': inds = _numpy.argsort(self.inc)
    elif key == 'omega': inds = _numpy.argsort(self.omega)
    elif key == 'Omega': inds = _numpy.argsort(self.Omega)
    else: raise InvalidValueForKeyException()

    self.m[:] = self.m[inds]
    self.b[:] = self.b[inds]
    self.v[:] = self.v[inds]
    self.inc[:] = self.inc[inds]
    self.omega[:] = self.omega[inds]
    self.Omega[:] = self.Omega[inds]

  @property
  def m(self):
    return self._m

  @property
  def mass(self):
    return self.m
  
  @property
  def b(self):
    return self._b

  @property
  def impact_parameter(self):
    return self.b

  @property
  def v(self):
    return self._v

  @property
  def velocity(self):
    return self.v

  @property
  def inc(self):
    return self._inc

  @property
  def inclination(self):
    return self.inc

  @property
  def argument_periastron(self):
    return self.omega

  @property
  def omega(self):
    return self._omega

  @property
  def Omega(self):
    return self._Omega

  @property
  def longitude_ascending_node(self):
    return self.Omega

  @property
  def params(self):
    return [self.m, self.b, self.v, self.inc, self.omega, self.Omega]
  
  @property
  def param_values(self):
    return _numpy.array([self.m, self.b, self.v, self.inc, self.omega, self.Omega])



class StellarEnvironment:
  '''
    This is the AIRBALL StellarEnvironment class.
    It encapsulates the relevant data for a static stellar environment.

    # Example
    my_env = airball.StellarEnvironment(stellar_density=10, velocity_dispersion=20, lower_mass_limit=0.08, upper_mass_limit=8, name='My Environment')
    my_star = my_env.random_star()

    If a `maximum_impact_parameter` is not given, AIRBALL attempts to estimate a reasonable one.
    There are predefined subclasses for the LocalNeighborhood, a generic OpenCluster, a generic GlobularCluster, and the Milky Way center GalacticBulge and GalacticCore.
  '''
  def __init__(self, stellar_density, velocity_dispersion, lower_mass_limit, upper_mass_limit, mass_function=None, maximum_impact_parameter=None, name=None, UNIT_SYSTEM=[], object_name=None):

    # Check to see if an stars object unit is defined in the given UNIT_SYSTEM and if the user defined a different name for the objects.
    objectUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(units.stars)]
    if objectUnit == [] and object_name is not None: UNIT_SYSTEM.append(units.def_unit(object_name, units.stars))
    self.units = UnitSystem(UNIT_SYSTEM)

    self.density = stellar_density
    self.velocity_dispersion = velocity_dispersion
    self.maximum_impact_parameter = maximum_impact_parameter

    self._upper_mass_limit = upper_mass_limit.to(self.units.units['mass']) if isQuantity(upper_mass_limit) else upper_mass_limit * self.units.units['mass']
    self._lower_mass_limit = lower_mass_limit.to(self.units.units['mass']) if isQuantity(lower_mass_limit) else lower_mass_limit * self.units.units['mass']
    self._IMF = IMF(min_mass=self._lower_mass_limit, max_mass=self._upper_mass_limit, mass_function=mass_function, unit=self.units.units['mass'])
    self._median_mass = self.IMF.median_mass

    self.name = name if name is not None else 'Stellar Environment'

  def random_star(self, maximum_impact_parameter=None, include_orientation=True, size=1):
    '''
      Computes a random star from a stellar environment.
      Returns: airball.Star() or airball.Stars() if size > 1.
    '''
    size = int(size)

    v = _maxwell.rvs(scale=_scale(self.velocity_dispersion), size=size) # Relative velocity of the star at infinity.

    max_impact = maximum_impact_parameter if maximum_impact_parameter is not None else self.maximum_impact_parameter
    b = max_impact * _numpy.sqrt(_uniform.rvs(size=size)) # Impact parameter of the star.
    
    m = self.IMF.random_mass(size=size) # Mass of the star.
    
    zeros = _numpy.zeros(size)
    inc = 2.0*_numpy.pi * _uniform.rvs(size=size) - _numpy.pi if include_orientation else zeros
    ω = 2.0*_numpy.pi * _uniform.rvs(size=size) - _numpy.pi if include_orientation else zeros
    Ω = 2.0*_numpy.pi * _uniform.rvs(size=size) - _numpy.pi if include_orientation else zeros

    if size > 1: return Stars(m=m, b=b, v=v, inc=inc, omega=ω, Omega=Ω)
    else: return Star(m, b[0], v[0], inc[0], ω[0], Ω[0])

  def stats(self):
    ''' 
    Prints a summary of the current stats of the Stellar Environment.
    '''
    s = self.name
    s += "\n------------------------------------------\n"
    s += "Stellar Density:     {0:12.4g} \n".format(self.density)
    s += "Velocity Scale:      {0:12.4g} \n".format(self.velocity_dispersion)
    s += "Mass Range:            {0:6.4g} - {1:1.4g}\n".format(self.lower_mass_limit.value, self.upper_mass_limit)
    s += "Median Mass:         {0:12.4g} \n".format(self.median_mass)
    s += "Max Impact Param:    {0:12.4g} \n".format(self.maximum_impact_parameter)
    s += "Encounter Rate:      {0:12.4g} \n".format(self.encounter_rate)
    s += "------------------------------------------"
    print(s)

  @property
  def object_unit(self):
    return self.units.units['object']
  
  @property
  def UNIT_SYSTEM(self):
      return self._UNIT_SYSTEM

  @UNIT_SYSTEM.setter
  def UNIT_SYSTEM(self, UNIT_SYSTEM):
    self.units.UNIT_SYSTEM = UNIT_SYSTEM

  @property
  def median_mass(self):
    '''
      The median mass of the environment's IMF
    '''
    return self.IMF.median_mass.to(self.units.units['mass'])

  @property
  def maximum_impact_parameter(self):
    '''
      Compute an estimate for the largest impact parameter to affect a Sun-Neptune system.
    '''
    # TODO: Update with UNITS
    if self._maximum_impact_parameter is None:
      sim = _rebound.Simulation()
      sim.add(m=1.0)
      sim.add(m=5.2e-05, a=30.2, e=0.013)
      _f = lambda b: _numpy.log10(_numpy.abs(1e-16 - _numpy.abs(relative_energy_change(sim, Star(self.upper_mass_limit, b, _numpy.sqrt(2.0)*_scale(self.velocity_dispersion))))))
      bs = _numpy.logspace(1, 6, 1000)
      b0 = bs[_numpy.argmin(_f(bs))]
      self._maximum_impact_parameter = _fminbound(_f, b0/5, 5*b0) * units.au
    return self._maximum_impact_parameter.to(self.units.units['length'])
  
  @maximum_impact_parameter.setter
  def maximum_impact_parameter(self, value):
    if value is not None:
      self._maximum_impact_parameter = value.to(self.units.units['length']) if isQuantity(value) else value * self.units.units['length']

  @property
  def density(self):
    '''
      The number density of the environment.
      Default units: pc^{-3}.
    '''
    return self._density.to(self.units.units['density'])

  @density.setter
  def density(self, value):
    '''
      The number density of the environment.
      Default units: pc^{-3}.
    '''
    if isQuantity(value):
      if value.unit.is_equivalent(units.stars/units.m**3): self._density = value.to(self.units.units['density'])
      elif value.unit.is_equivalent(1/units.m**3): self._density = (value * self.units.units['object']).to(self.units.units['density'])
      else: AssertionError('The given density units are not compatible.')
    else: self._density = value * self.units.units['density']

  @property
  def velocity_dispersion(self):
    '''
      Return the velocity dispersion of the environment.
      Default units: km/s.
    '''
    return self._velocity.to(self.units.units['velocity'])
  
  @velocity_dispersion.setter
  def velocity_dispersion(self, value):
    '''
      The velocity dispersion of the environment.
      Default units: km/s.
    '''
    self._velocity = value.to(self.units.units['velocity']) if isQuantity(value) else value * self.units.units['velocity']

  @property
  def lower_mass_limit(self):
    '''
      Return the lower mass limit of the IMF of the environment.
      Default units: solMass
    '''
    return self.IMF.min_mass.to(self.units.units['mass'])
  
  @lower_mass_limit.setter
  def lower_mass_limit(self, value):
    '''
      The lower mass limit of the IMF of the environment.
      Default units: solMass
    '''
    self.IMF.min_mass = value

  @property
  def upper_mass_limit(self):
    '''
      Return the lower mass limit of the IMF of the environment.
      Default units: solMass
    '''
    return self.IMF.max_mass.to(self.units.units['mass'])
  
  @upper_mass_limit.setter
  def upper_mass_limit(self, value):
    '''
      The lower mass limit of the IMF of the environment.
      Default units: solMass
    '''
    self.IMF.max_mass = value


  @property
  def IMF(self):
    '''
      Return the IMF of the environment.
    '''
    return self._IMF
  
  @IMF.setter
  def IMF(self, value):
    '''
      The IMF of the environment.
    '''
    if isinstance(value, IMF): self._IMF = value
    else: raise AssertionError('Initial Mass Function (IMF) must be an airball.IMF object.')
  
  @property
  def encounter_rate(self):
    '''
        Compute the expected flyby encounter rate Γ = ⟨nσv⟩ for the stellar environment in units of flybys per year.
        The inverse of the encouter rate will give the average number of years until a flyby.

        n : stellar number density. Default units: pc^{-3}
        σ : interaction cross section. Default units: AU^2
        v : velocity dispersion. Default units: km/s

        The interaction cross section σ = πb^2 considers gravitational focussing b = q√[1 + (2GM)/(q v∞^2)] and considers
        - the median mass of the environment
        - the maximum impact parameter
        - the relative velocity at infinity derived from the velocity dispersion
    '''
    return encounter_rate(self._density, _scale(self.velocity_dispersion), self._maximum_impact_parameter, star_mass=self.median_mass).to(self.units.units['object']/self.units.units['time'])



class LocalNeighborhood(StellarEnvironment):
  '''
    This is a AIRBALL StellarEnvironment subclass for the Local Neighborhood.
    It encapsulates the relevant data for a static stellar environment representing the local neighborhood of the solar system.

    The stellar density is 0.14 pc^-3 defined by Bovy (2017).
    The velocity scale is 26 km/s, defined by Bailer-Jones et al. (2018) so that 90% of stars have v < 100 km/s.
    The mass limit is defined to between 0.01-10 solar masses using Chabrier (2003) for single stars when m < 1 and Salpeter (1955) for stars m ≥ 1.

    # Example
    my_local = airball.LocalNeighborhood()
    my_10stars = my_local.random_star(size=10)
    # returns a Stars object with the masses, impact parameters, velocities, and orientation of the 10 Star objects in a heliocentric model.
  '''
  short_name = 'Local'
  def __init__(self, stellar_density = 0.14 * units.stars/units.pc**3, velocity_dispersion = 22 * units.km/units.s, lower_mass_limit=0.08 * units.solMass, upper_mass_limit = 8 * units.solMass, maximum_impact_parameter=None, UNIT_SYSTEM=[], mass_function=None):
    super().__init__(stellar_density=stellar_density, velocity_dispersion=velocity_dispersion, lower_mass_limit=lower_mass_limit, upper_mass_limit=upper_mass_limit, mass_function=mass_function, maximum_impact_parameter=maximum_impact_parameter, UNIT_SYSTEM=UNIT_SYSTEM, name = 'Local Neighborhood')
    self.maximum_impact_parameter = 10000 * units.au if maximum_impact_parameter is None else maximum_impact_parameter

class OpenCluster(StellarEnvironment):
  '''
    This is a AIRBALL StellarEnvironment subclass for a generic Open Cluster.
    It encapsulates the relevant data for a static stellar environment representing a generic open cluster.

    The stellar density is 100 pc^-3 informed by Adams (2010).
    The velocity scale is 1 km/s informed by Adams (2010) and Malmberg et al. (2011).
    The mass limit is defined to between 0.01-100 solar masses using Chabrier (2003) for single stars when m < 1 and Salpeter (1955) for stars m ≥ 1.

    # Example
    my_open = airball.OpenCluster()
    my_10stars = my_open.random_star(size=10)
    # returns a Stars object with the masses, impact parameters, velocities, and orientation of the 10 Star objects in a heliocentric model.
  '''
  short_name = 'Open'
  
  def __init__(self, maximum_impact_parameter=None, UNIT_SYSTEM=[]):
    super().__init__(stellar_density = 100 * units.stars * units.pc**-3, velocity_dispersion = 1 * units.km/units.s, lower_mass_limit=0.08 * units.solMass, upper_mass_limit = 100 * units.solMass, maximum_impact_parameter=maximum_impact_parameter, UNIT_SYSTEM=UNIT_SYSTEM, name = 'Open Cluster')
    self._maximum_impact_parameter = 1000 * units.au

class GlobularCluster(StellarEnvironment):
  short_name = 'Globular'
  
  def __init__(self, maximum_impact_parameter=None, UNIT_SYSTEM=[]):
    super().__init__(stellar_density = 1000 * units.stars * units.pc**-3, velocity_dispersion = 10 * units.km/units.s, lower_mass_limit=0.08 * units.solMass, upper_mass_limit = 1 * units.solMass, maximum_impact_parameter=maximum_impact_parameter, UNIT_SYSTEM=UNIT_SYSTEM, name = 'Globular Cluster')
    self._maximum_impact_parameter = 5000 * units.au

class GalacticBulge(StellarEnvironment):
  short_name = 'Bulge'
  
  def __init__(self, maximum_impact_parameter=None, UNIT_SYSTEM=[]):
    super().__init__(stellar_density = 50 * units.stars * units.pc**-3, velocity_dispersion = 120 * units.km/units.s, lower_mass_limit=0.08 * units.solMass, upper_mass_limit = 10 * units.solMass, maximum_impact_parameter=maximum_impact_parameter, UNIT_SYSTEM=UNIT_SYSTEM, name = 'Milky Way Bulge')
    self._maximum_impact_parameter = 50000 * units.au

class GalacticCore(StellarEnvironment):
  short_name = 'Core'
  
  def __init__(self, maximum_impact_parameter=None, UNIT_SYSTEM=[]):
    super().__init__(stellar_density = 10000 * units.stars * units.pc**-3, velocity_dispersion = 170 * units.km/units.s, lower_mass_limit=0.08 * units.solMass, upper_mass_limit = 10 * units.solMass, maximum_impact_parameter=maximum_impact_parameter, UNIT_SYSTEM=UNIT_SYSTEM, name = 'Milky Way Core')
    self._maximum_impact_parameter = 50000 * units.au


################################
###### Custom Exceptions #######
################################

class InvalidStarException(Exception):
  def __init__(self): super().__init__('Object is not a valid airball.Star object.')

class StarInvalidKeyException(Exception):
  def __init__(self): super().__init__('Invalid key for Stars.')

class UnspecifiedParameterException(Exception):
  def __init__(self, message): super().__init__(message)

class OverspecifiedParametersException(Exception):
  def __init__(self, message): super().__init__(message)

class ListLengthException(Exception):
  def __init__(self): super().__init__('List arguments must be same length.')

class IncompatibleListException(Exception):
  def __init__(self): super().__init__('The given list type is not compatible. Please use a Python list or an ndarray.')

class InvalidValueForKeyException(Exception):
  def __init__(self): super().__init__('Invalid value for key.')