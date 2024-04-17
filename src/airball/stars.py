import numpy as _np
import rebound as _rebound
import pickle as _pickle
from scipy.stats import uniform as _uniform
from . import environments as _env
from . import tools as _tools
from . import units as _u

try: from collections.abc import MutableMapping # Required for Python>=3.9
except: from collections import MutableMapping

class Star:
  '''
    This is the AIRBALL Star class.
    It encapsulates the relevant parameters for a given star. Only the mass is a quantity intrinsic to the object. The impact parameter, velocity, inclination, argument of periastron, and longitude of the ascending node quantities are defined with respect to the host star and plane passing through the star.

    Args:
      m (float): The mass of the star. Default units are in solMass.
      b (float): The impact parameter of the star. Default units are in AU. 
      v (float): The velocity at infinity of the star. Default units are in km/s.
      inc (float, optional): The inclination of the star. Default units are in radians.
      omega (float, optional): The argument of the periastron of the star. Default units are in radians.
      Omega (float, optional): The longitude of the ascending node of the star. Default units are in radians.
      UNIT_SYSTEM (list, optional): The unit system to use for the parameters. Default is [u.solMass, u.AU, u.km/u.s, u.rad].

    Attributes:
      m (astropy.units.Quantity): The mass of the star. Default units are in solMass.
      b (astropy.units.Quantity): The impact parameter of the star. Default units are in AU. 
      v (astropy.units.Quantity): The velocity at infinity of the star. Default units are in km/s.
      inc (astropy.units.Quantity): The inclination of the star. Default units are in radians.
      omega (astropy.units.Quantity): The argument of the periastron of the star. Default units are in radians.
      Omega (astropy.units.Quantity): The longitude of the ascending node of the star. Default units are in radians.
      units (airball.tools.UnitSet): The unit system to use for the parameters. Default is [u.solMass, u.AU, u.km/u.s, u.rad].
      impulse_gradient (astropy.units.Quantity): The impulse gradient of the star. Default units are in km/s/AU.
      params (list): A list of the parameters of the star in order of: Mass, m; Impact Parameter, b; Velocity, v; Inclination, inc; Argument of the Periastron, omega; and Longitude of the Ascending Node, Omega.
      param_values (list): A list of the values of the parameters of the star in order of: Mass, m; Impact Parameter, b; Velocity, v; Inclination, inc; Argument of the Periastron, omega; and Longitude of the Ascending Node, Omega.

    Examples:
      ```python
      import airball
      star = airball.Star(m=1.0, b=1.0, v=1.0, inc=0.0, omega=0.0, Omega=0.0)
      ```

      ```python
      import airball
      star = airball.Star(m=1.0, b=1.0, v=1.0, inc='uniform', omega='uniform', Omega='uniform')
      ```

      ```python
      import airball
      star = airball.Star(m=1.0, b=1.0, v=1.0)
      ```
  '''
  def __init__(self, m, b, v, inc='uniform', omega='uniform', Omega='uniform', UNIT_SYSTEM=[], **kwargs) -> None:
    self.units = _u.UnitSet(UNIT_SYSTEM)

    if inc == 'uniform' or inc == None: inc = 2.0*_np.pi * _uniform.rvs() - _np.pi
    if omega == 'uniform' or omega == None: omega = 2.0*_np.pi * _uniform.rvs() - _np.pi
    if Omega == 'uniform' or Omega == None: Omega = 2.0*_np.pi * _uniform.rvs() - _np.pi

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
  def N(self): return 1

  @property
  def m(self):
    return self._mass.to(self.units['mass'])

  @m.setter
  def m(self, value):
    self._mass = _tools.verify_unit(value, self.units['mass'])

  @property
  def mass(self):
    return self.m

  @mass.setter
  def mass(self, value):
    self.m = value
  
  @property
  def b(self):
    return self._impact_parameter.to(self.units['length'])

  @b.setter
  def b(self, value):
    self._impact_parameter = _tools.verify_unit(value, self.units['length'])

  @property
  def impact_parameter(self):
    return self.b

  @impact_parameter.setter
  def impact_parameter(self, value):
    self.b = value

  @property
  def v(self):
    return self._velocity.to(self.units['velocity'])

  @v.setter
  def v(self, value):
    self._velocity = _tools.verify_unit(value, self.units['velocity'])

  @property
  def velocity(self):
    return self.v

  @velocity.setter
  def velocity(self, value):
    self.v = value

  @property
  def inc(self):
    return self._inclination.to(self.units['angle'])

  @inc.setter
  def inc(self, value):
    self._inclination = _tools.verify_unit(value, self.units['angle'])

  @property
  def inclination(self):
    return self.inc

  @inc.setter
  def inclination(self, value):
    self.inc = value

  @property
  def omega(self):
    return self._argument_periastron.to(self.units['angle'])

  @omega.setter
  def omega(self, value):
    self._argument_periastron = _tools.verify_unit(value, self.units['angle'])

  @property
  def argument_periastron(self):
    return self.omega

  @argument_periastron.setter
  def argument_periastron(self, value):
    self.omega = value

  @property
  def Omega(self):
    return self._longitude_ascending_node.to(self.units['angle'])

  @Omega.setter
  def Omega(self, value):
    self._longitude_ascending_node = _tools.verify_unit(value, self.units['angle'])

  @property
  def longitude_ascending_node(self):
    return self.Omega

  @longitude_ascending_node.setter
  def longitude_ascending_node(self, value):
    self.Omega = value

  @property
  def impulse_gradient(self):
    # Calculate the impulse gradient for a flyby star.
    G = (1 * _u.au**3 / _u.solMass / _u.yr2pi**2)
    return ((2.0 * G * self.m) / (self.v * self.b**2.0)).to(_u.km/_u.s/_u.au)

  @property
  def params(self):
    # Returns a list of the parameters of the Stars (with units) in order of:
    # Mass, m; Impact Parameter, b; Velocity, v; Inclination, inc; Argument of the Periastron, omega; and Longitude of the Ascending Node, Omega
    return [self.m, self.b, self.v, self.inc, self.omega, self.Omega]
  
  @property
  def param_values(self):
    # Returns a list of the parameters of the Stars in order of:
    # Mass, m; Impact Parameter, b; Velocity, v; Inclination, inc; Argument of the Periastron, omega; and Longitude of the Ascending Node, Omega
    return _np.array([self.m.value, self.b.value, self.v.value, self.inc.value, self.omega.value, self.Omega.value])
   
  def eccentricity(self, sim):
    return self.e(sim)

  def e(self, sim):
    sim_units = _tools.rebound_units(sim)
    G = (sim.G * sim_units['length']**3 / sim_units['mass'] / sim_units['time']**2)
    mu = G * (_tools.system_mass(sim) * sim_units['mass'] + self.m)

    numerator = self.b * self.v*self.v
    return _np.sqrt(1 + (numerator/mu)**2.)
  
  def periastron(self, sim):
    return self.q(sim)

  def q(self, sim):
    sim_units = _tools.rebound_units(sim)
    G = (sim.G * sim_units['length']**3 / sim_units['mass'] / sim_units['time']**2)
    mu = G * (_tools.system_mass(sim)  * sim_units['mass'] + self.m)

    numerator = self.b * self.v*self.v
    star_e = _np.sqrt(1 + (numerator/mu)**2.)
    return self.b * _np.sqrt((star_e - 1.0)/(star_e + 1.0))
 
  def stats(self, returned=False):
    # Prints a summary of the current stats of the Star.
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
    return NotImplemented
  
  def __eq__(self, other):
    # Overrides the default implementation
    if isinstance(other, Star):
        data = ((self.m == other.m) and (self.b == other.b) and (self.v == other.v) and (self.inc == other.inc) and (self.omega == other.omega) and (self.Omega == other.Omega))
        properties = (self.units == other.units)
        return data and properties
    return NotImplemented

  def __hash__(self):
    # Overrides the default implementation.
    data = []
    for d in sorted(self.__dict__.items()):
        try: data.append((d[0], tuple(d[1])))
        except: data.append(d)
    data = tuple(data)
    return hash(data)


class Stars(MutableMapping):
  '''
    This class allows the user to access stars like an array using the star's index. 
    Allows for negative indices and slicing.
    The implementation uses astropy.Quantity and numpy ndarrays underneath and only generates a airball.Star object when a single Star is requested.

    Args:
      m (list, ndarray, or Quantity): The masses of the stars. Default units are in solMass.
      b (list, ndarray, or Quantity): The impact parameters of the stars. Default units are in AU.
      v (list, ndarray, or Quantity): The velocities at infinity of the stars. Default units are in km/s.
      inc (list, ndarray, or Quantity, optional): The inclinations of the stars. Default units are in radians.
      omega (list, ndarray, or Quantity, optional): The arguments of the periastron of the stars. Default units are in radians.
      Omega (list, ndarray, or Quantity, optional): The longitudes of the ascending node of the stars. Default units are in radians.
      UNIT_SYSTEM (list, optional): The unit system to use for the parameters. Default is [u.solMass, u.AU, u.km/u.s, u.rad].
      size (int, optional): The number of stars to generate. Default is None.
      environment (airball.Environment, optional): The environment to generate the stars in (if `size` > 1). `env` is an alias. Default is None.
      filename (str, optional): The name of the file to load the instance from. The file should be in binary format. Default is None.

    Attributes:
      m (astropy.units.Quantity): The masses of the stars. Default units are in solMass.
      b (astropy.units.Quantity): The impact parameters of the stars. Default units are in AU.
      v (astropy.units.Quantity): The velocities at infinity of the stars. Default units are in km/s.
      inc (astropy.units.Quantity): The inclinations of the stars. Default units are in radians.
      omega (astropy.units.Quantity): The arguments of the periastron of the stars. Default units are in radians.
      Omega (astropy.units.Quantity): The longitudes of the ascending node of the stars. Default units are in radians.
      units (airball.tools.UnitSet): The unit system to use for the parameters. Default is [u.solMass, u.AU, u.km/u.s, u.rad].
      median_mass (astropy.units.Quantity): The median mass of the stars. Default units are in solMass.
      mean_mass (astropy.units.Quantity): The mean mass of the stars. Default units are in solMass.
      N (int): The number of stars.
      shape (tuple): The shape of the stars array.
      params (list): A list of the parameters of the stars in order of: Mass, m; Impact Parameter, b; Velocity, v; Inclination, inc; Argument of the Periastron, omega; and Longitude of the Ascending Node, Omega.
      param_values (list): A list of the values of the parameters of the stars in order of: Mass, m; Impact Parameter, b; Velocity, v; Inclination, inc; Argument of the Periastron, omega; and Longitude of the Ascending Node, Omega.

    Examples:
      ```python
      # Explicitly specify the stellar parameters.
      import airball
      stars_from_params1 = airball.Stars(m=[1.0, 1.0, 1.0], b=[1.0, 1.0, 1.0], v=[1.0, 1.0, 1.0], inc=[0.0, 0.0, 0.0], omega=[0.0, 0.0, 0.0], Omega=[0.0, 0.0, 0.0])
      stars_from_params2 = airball.Stars(m=[1.0, 1.0, 1.0], b=[1.0, 1.0, 1.0], v=[1.0, 1.0, 1.0]) # random orientation angles
      stars_from_params3 = airball.Stars(m=[1.0, 1.0, 1.0], b=[1.0, 1.0, 1.0], v=[1.0, 1.0, 1.0], inc='uniform', omega='uniform', Omega='uniform') # random orientation angles
      stars_from_params4 = airball.Stars(m=1, b=200, v=5, omega=0, Omega=0, size=100) # 100 identical stars with random inclinations
      ```

      ```python
      # Randomly generate the stellar parameters from a given environment.
      import airball
      stars_from_env1 = airball.Stars(environment=airball.OpenCluster(), size=3)
      stars_from_env2 = airball.Stars(env=airball.LocalNeighborhood(), size=5)
      stars_from_env3 = airball.Stars(airball.GlobularCluster(), size=9)
      ```

      ```python
      # Load the stellar parameters from a file.
      import airball
      stars_from_file1 = airball.Stars(filename='open_cluster.stars')
      stars_from_file2 = airball.Stars('open_cluster.stars')
      ```
  '''
  def __init__(self, filename=None, **kwargs) -> None:
    try: 
      self.units = _u.UnitSet(kwargs['UNIT_SYSTEM'])
      del kwargs['UNIT_SYSTEM']
    except KeyError:
      try: self.units = kwargs['units']
      except KeyError: 
        self.units = _u.UnitSet()

    # Initialize Stars from file.
    if filename is not None and isinstance(filename, str):
      try:
        loaded = Stars._load(filename)
        self.__dict__ = loaded.__dict__
      except: raise Exception('Invalid filename.')
      return
    
    # If nothing is specified, return an empty Stars object.
    if not kwargs: 
      self._m = _np.array([]) << self.units['mass']
      self._b = _np.array([]) << self.units['length']
      self._v = _np.array([]) << self.units['velocity']
      self._inc = _np.array([]) << self.units['angle']
      self._omega = _np.array([]) << self.units['angle']
      self._Omega = _np.array([]) << self.units['angle']
      self._shape = (0,)
      self.environment = None
      return

    # Determine the number of stars to generate.
    self._shape = (0,)
    for key in kwargs:
      try:
        if isinstance(kwargs[key], _u.Quantity):
            shape = kwargs[key].shape
            N = 0
            if len(shape) >= 1: N = _np.prod(shape) 
            else: shape = (0,)
            if N > self.N: self._shape = shape
        elif _tools.isList(kwargs[key]):
          if _np.prod(_np.shape(kwargs[key])) > self.N: self._shape = _np.shape(kwargs[key])
        else: pass
      except TypeError as err:
        print(err)
        self._shape = (len(kwargs[key]),)
      except Exception as err: raise err

    if 'size' in kwargs and self.N != 0: raise OverspecifiedParametersException('If lists are given then size cannot be specified.')
    elif 'size' in kwargs:
      if isinstance(kwargs['size'], tuple): 
        self._shape = tuple([int(i) for i in kwargs['size']])
      else: self._shape = (int(kwargs['size']),)
    elif self.N == 0: raise UnspecifiedParameterException('If no lists of parameters are given then size must be specified.')
    else: pass # No errors or issues, continue.


    # Initialize Stars from environment.
    self.environment = kwargs.get('environment', None)
    if (('environment' in kwargs or 'env' in kwargs) or  isinstance(filename, _env.StellarEnvironment)) and 'size' in kwargs:
      if self.N <= 1: raise InvalidValueForKeyException("If a generating environment is given then size must be greater than 1.")
      if 'environment' in kwargs:
        se = kwargs['environment'] 
        del kwargs['environment']
      elif 'env' in kwargs:
        se = kwargs['env']
        del kwargs['env']
      else:
        se = filename
        del filename
      stars = se.random_stars(**kwargs)
      self.__dict__ = stars.__dict__
      return

    # Initialize Stars from kwargs.
    keys = ['m', 'b', 'v']
    units = ['mass', 'length', 'velocity']
    unspecifiedParameterExceptions = ['Mass, m, must be specified.', 'Impact Parameter, b, must be specified.', 'Velocity, v, must be specified.']

    for k,u,upe in zip(keys, units, unspecifiedParameterExceptions):
      try:
        # Check to see if was key is given.
        value = kwargs[k]
        # Check if shape matches other key values.
        if len(value) != len(self): raise ListLengthException(f'Difference of {len(value)} and {len(self)} for {k}.')
        # Length of value matches other key values, check if value is a list.
        elif isinstance(value, list):
          # Value is a list, try to turn list of Quantities into a ndarray Quantity.
          try: quantityValue = _np.array([v.to(self.units[u]).value for v in value]) << self.units[u]
          # Value was not a list of Quantities, turn list into ndarray and make a Quantity.
          except: quantityValue = _np.array(value) << self.units[u]
        # Value was not a list, check to see if value is an ndarray.
        elif isinstance(value, _np.ndarray):
          # Assume ndarray is a Quantity and try to convert ndarray into given units.
          try: quantityValue = value.to(self.units[u])
          # ndarray is not a Quantity so turn ndarray into a Quantity.
          except: quantityValue = value << self.units[u]
        # Value implements __len__, but is not a list or ndarray.
        else: raise IncompatibleListException()
      # This key is necessary and must be specified, raise and Exception.
      except KeyError: raise UnspecifiedParameterException(upe)
      # Value is not a list, so assume it is an int or float and generate an ndarray of the given value.
      except TypeError: 
        value = value.to(self.units[u]) if _tools.isQuantity(value) else value * self.units[u]
        quantityValue = _np.ones(self.shape) * value
      # Catch any additional Exceptions.
      except Exception as err: raise err
      # Store Quantity Value as class property.
      if k == 'm': self._m = quantityValue
      elif k == 'b': self._b = quantityValue
      elif k == 'v': self._v = quantityValue
      else: raise _tools.InvalidKeyException()
      # Double check for consistent shapes.
      if self._shape is not None:
        if quantityValue.shape != self._shape: raise ListLengthException(f'Difference of {quantityValue.shape} and {self._shape} for {k}.')
      else: self._shape = quantityValue.shape
        
    for k in ['inc', 'omega', 'Omega']:
      try:
        # Check to see if was key is given.
        value = kwargs[k]
          # Check to see if value for key is string.
        if isinstance(value, str):
          # Value is a string, check to see if value for key is valid.
          if value != 'uniform': raise InvalidValueForKeyException()
          # Value 'uniform' for key is valid, now generate an array of values for key.
          quantityValue = (2.0*_np.pi * _uniform.rvs(size=self.shape) - _np.pi) * self.units['angle']
        # Value is not a string, check if length matches other key values.
        elif _np.shape(value) != self.shape: raise ListLengthException(f'Difference of {_np.shape(value)} and {self.shape} for {k}.')
        # Length of value matches other key values, check if value is a list.
        elif isinstance(value, list):
          # Value is a list, try to turn list of Quantities into a ndarray Quantity.
          try: quantityValue = _np.array([v.to(self.units['angle']).value for v in value]) * self.units['angle']
          # Value was not a list of Quantities, turn list into ndarray and make a Quantity.
          except: quantityValue = _np.array(value) * self.units['angle']
        # Value was not a list, check to see if value is an ndarray.
        elif isinstance(value, _np.ndarray):
          # Assume ndarray is a Quantity and try to convert ndarray into given units.
          try: quantityValue = value.to(self.units['angle'])
          # ndarray is not a Quantity so turn ndarray into a Quantity.
          except: quantityValue = value * self.units['angle']
        # Value implements __len__, but is not a list or ndarray.
        else: raise IncompatibleListException()
      # Key does not exist, assume the user wants an array of values to automatically be generated.
      except KeyError: 
        quantityValue = (2.0*_np.pi * _uniform.rvs(size=self.shape) - _np.pi) * self.units['angle']
      # Value is not a list, so assume it is an int or float and generate an ndarray of the given value.
      except TypeError: 
        value = value.to(self.units['angle']) if _tools.isQuantity(value) else value * self.units['angle']
        quantityValue = _np.ones(self.shape) * value
      # Catch any additional Exceptions.
      except Exception as err: raise err
      # Store Quantity Value as class property.
      if k == 'inc': self._inc = quantityValue
      elif k == 'omega': self._omega = quantityValue
      elif k == 'Omega': self._Omega = quantityValue
      else: raise _tools.InvalidKeyException()
      # Double check for consistent shapes.
      if self.shape is not None:
        if quantityValue.shape != self.shape: raise ListLengthException(f'Difference of {quantityValue.shape} and {self.shape} for {k}.')
      else: self._shape = quantityValue.shape

  @property
  def N(self):
    '''The total number of Stars.'''
    return _np.prod(self.shape)
    
  @property
  def shape(self):
    '''The shape of the Stars arrays.'''
    return self._shape

  @property
  def median_mass(self):
    '''The median mass of the Stars.'''
    return _np.median([mass.value for mass in self.m]) << self.units['mass']
  
  @property
  def mean_mass(self):
    '''The mean mass of the Stars.'''
    return _np.mean([mass.value for mass in self.m]) << self.units['mass']
  
  @property
  def m(self):
    return self._m << self.units['mass']
  
  @m.setter
  def m(self, value):
    try: 
      if _np.shape(value) != self.shape:  raise ListLengthException(f'Difference of {_np.shape(value)} and {self.shape}.')
    except (TypeError, AttributeError): raise IncompatibleListException()
    self._m = _tools.verify_unit(value, self.units['mass'])

  @property
  def mass(self):
    """
    Set the masses of the Stars, alias of `m` (default units: Msun).

    Args:
      value (list, ndarray, or Quantity): The new mass values. Can be set with a list such that len(value) == len(Stars).

    Raises:
      ListLengthException: If the length of the provided list does not match the number of Stars.
      IncompatibleListException: If the provided list is not compatible.
    """
    return self.m
  
  @mass.setter
  def mass(self, value):
    try: self.m = value
    except Exception as err: raise err
  
  @property
  def b(self):
    return self._b << self.units['length']
  
  @b.setter
  def b(self, value):
    try: 
      if _np.shape(value) != self.shape:  raise ListLengthException(f'Difference of {_np.shape(value)} and {self.shape}.')
    except (TypeError, AttributeError): raise IncompatibleListException()
    self._b = _tools.verify_unit(value, self.units['length'])

  @property
  def impact_parameter(self):
    """
    Set the impact parameters of the Stars, alias of `b` (default units: AU).

    Args:
      value (list, ndarray, or Quantity): The new impact parameter values. Can be set with a list such that len(value) == len(Stars).

    Raises:
      ListLengthException: If the length of the provided list does not match the number of Stars.
      IncompatibleListException: If the provided list is not compatible.
    """
    return self.b
  
  @impact_parameter.setter
  def impact_parameter(self, value):
    try: self.b = value
    except Exception as err: raise err

  @property
  def v(self):
    return self._v << self.units['velocity']

  @v.setter
  def v(self, value):
    try: 
      if _np.shape(value) != self.shape:  raise ListLengthException(f'Difference of {_np.shape(value)} and {self.shape}.')
    except (TypeError, AttributeError): raise IncompatibleListException()
    self._v = _tools.verify_unit(value, self.units['velocity'])

  @property
  def velocity(self):
    '''
    Set the velocities at infinity of the Stars, alias of `v` (default units: km/s).

    Args:
      value (list, ndarray, or Quantity): The new velocity values. Can be set with a list such that len(value) == len(Stars).

    Raises:
      ListLengthException: If the length of the provided list does not match the number of Stars.
      IncompatibleListException: If the provided list is not compatible.
    '''
    return self.v
  
  @velocity.setter
  def velocity(self, value):
    try: self.v = value
    except Exception as err: raise err

  @property
  def inc(self):
    return self._inc << self.units['angle']

  @inc.setter
  def inc(self, value):
    try: 
      if _np.shape(value) != self.shape:  raise ListLengthException(f'Difference of {_np.shape(value)} and {self.shape}.')
    except (TypeError, AttributeError): raise IncompatibleListException()
    self._inc = _tools.verify_unit(value, self.units['angle'])

  @property
  def inclination(self):
    '''
    The inclinations of the Stars, alias of `inc` (default units: radians).

    Args:
      value (list, ndarray, or Quantity): The new inclination values. Can be set with a list such that len(value) == len(Stars).

    Raises:
      ListLengthException: If the length of the provided list does not match the number of Stars.
      IncompatibleListException: If the provided list is not compatible.
    '''
    return self.inc
  
  @inclination.setter
  def inclination(self, value):
    try: self.inc = value
    except Exception as err: raise err

  @property
  def omega(self):
    return self._omega << self.units['angle']

  @omega.setter
  def omega(self, value):
    try: 
      if _np.shape(value) != self.shape:  raise ListLengthException(f'Difference of {_np.shape(value)} and {self.shape}.')
    except (TypeError, AttributeError): raise IncompatibleListException()
    self._omega = _tools.verify_unit(value, self.units['angle'])

  @property
  def argument_periastron(self):
    '''
    The argument of periastron of the Stars, alias of `omega` (default units: radians).

    Args:
      value (list, ndarray, or Quantity): The new values for the argument of periastron. Can be set with a list such that len(value) == len(Stars).

    Raises:
      ListLengthException: If the length of the provided list does not match the number of Stars.
      IncompatibleListException: If the provided list is not compatible.
    '''
    return self.omega

  @argument_periastron.setter
  def argument_periastron(self, value):
    try: self.omega = value
    except Exception as err: raise err

  @property
  def Omega(self):
    return self._Omega << self.units['angle']

  @Omega.setter
  def Omega(self, value):
    try: 
      if _np.shape(value) != self.shape:  raise ListLengthException(f'Difference of {_np.shape(value)} and {self.shape}.')
    except (TypeError, AttributeError): raise IncompatibleListException()
    self._Omega = _tools.verify_unit(value, self.units['angle'])

  @property
  def longitude_ascending_node(self):
    '''
    The longitude of the ascending node of the Stars, alias of `Omega` (default units: radians).

    Args:
      value (list, ndarray, or Quantity): The new longitude of the ascending node values. Can be set with a list such that len(value) == len(Stars).

    Raises:
      ListLengthException: If the length of the provided list does not match the number of Stars.
      IncompatibleListException: If the provided list is not compatible.
    '''
    return self.Omega
  
  @longitude_ascending_node.setter
  def longitude_ascending_node(self, value):
    try: self.Omega = value
    except Exception as err: raise err

  @property
  def impulse_gradient(self):
    '''
      Calculate the impulse gradient for a flyby star.
      
      $$\\frac{2GM_\\star}{V_\\star b_\\star^2}$$  
    '''
    G = (1 * _u.au**3 / _u.solMass / _u.yr2pi**2)
    return ((2.0 * G * self.m) / (self.v * self.b**2.0)).to(_u.km/_u.s/_u.au)

  @property
  def params(self):
    '''
      A list of the parameters of the Stars (with units) in order of: Mass, m; Impact Parameter, b; Velocity, v; Inclination, inc; Argument of the Periastron, omega; and Longitude of the Ascending Node, Omega

      `[m, b, v, inc, omega, Omega]`
    '''
    return [self.m, self.b, self.v, self.inc, self.omega, self.Omega]
  
  @property
  def param_values(self):
    '''
      A list of the values of the parameters of the Stars in order of: Mass, m; Impact Parameter, b; Velocity, v; Inclination, inc; Argument of the Periastron, omega; and Longitude of the Ascending Node, Omega

      `[m.value, b.value, v.value, inc.value, omega.value, Omega.value]`
    '''
    return _np.array([self.m.value, self.b.value, self.v.value, self.inc.value, self.omega.value, self.Omega.value])
 
  def eccentricity(self, sim):
    '''
    The eccentricity of the Stars, alias for `e(sim)`.

    Args:
      sim (Simulation): The REBOUND Simulation to use for calculating the eccentricity.

    Returns:
      eccentricity (ndarray): The eccentricity of the Stars.
    '''
    return self.e(sim)

  def e(self, sim):
    units = _tools.rebound_units(sim)
    G = (sim.G * units.length**3 / units.mass / units.time**2)
    mu = G * (_tools.system_mass(sim) * units.mass + self.m)

    numerator = self.b * self.v*self.v
    return _np.sqrt(1 + (numerator/mu)**2.)
  
  def periastron(self, sim):
    '''
    The periastron of the Stars, alias for `q(sim)`.

    Args:
      sim (Simulation): The REBOUND Simulation to use for calculating the periastron.

    Returns:
      periastron (ndarray): The periastron of the Stars.
    '''
    return self.q(sim)

  def q(self, sim):
    sim_units = _tools.rebound_units(sim)
    G = (sim.G * sim_units['length']**3 / sim_units['mass'] / sim_units['time']**2)
    mu = G * (_tools.system_mass(sim)  * sim_units['mass'] + self.m)

    numerator = self.b * self.v*self.v
    star_e = _np.sqrt(1 + (numerator/mu)**2.)
    return self.b * _np.sqrt((star_e - 1.0)/(star_e + 1.0))
  
  def copy(self):
    '''Returns a deep copy of the Stars.'''
    newstars = self[:]
    newstars.environment = self.environment.copy()
    return newstars

  def sort(self, key, sim=None, argsort=False):
    # Alias for `sortby`.
    return self.sortby(key, sim, argsort)
  
  def argsort(self, key, sim=None):
    # Alias for `sortby(argsort=True)`.
    return self.sortby(key, sim, argsort=True)
  
  def argsortby(self, key, sim=None):
    # Alias for `sortby(argsort=True)`.
    return self.sortby(key, sim, argsort=True)

  def sortby(self, key, sim=None, argsort=False):
    """
    Sort the Stars in ascending order by a defining parameter.

    Args:
      key (str): The parameter to sort by. Options include: 'm' (mass), 'b' (impact parameter), 'v' (relative velocity at infinity), 'inc' (inclination), 'omega' (argument of the periastron), 'Omega' (longitude of the ascending node), 'q' (periapsis), 'e' (eccentricity). For 'q' and 'e', a REBOUND Simulation is required.
      sim (Simulation, optional): The REBOUND Simulation to use for sorting by 'q' and 'e'. Default is None.
      argsort (bool, optional): If True, return the indices used to sort the Stars. Default is False.

    Returns:
      indices (list or None): The indices used to sort the Stars, if argsort is True. Otherwise, None and sorting is done in place.

    !!! Info
        The Stars can also be sorted arbitrarily by providing a list of indices of length len(stars) as the key.

    Examples:
      ```python
      import airball
      stars = airball.Stars(m=[1, 2, 3], b=[3,2,1], v=[6,5,7])
      stars.sortby('b')
      inds  = stars.sortby('v', argsort=True)
      ```

      ```python
      import airball
      import rebound
      sim = rebound.Simulation()
      sim.add(m=1)
      sim.add(m=5e-5, a=30, e=0.01)
      stars = airball.Stars(airball.OpenCluster(), size=100)
      stars.sortby('q', sim=sim)
      ```   

      ```python
      import airball
      stars = airball.Stars(airball.LocalNeighborhood(), size=5)
      inds = [0,4,1,3,2]
      stars.sortby(inds)
      ```      
    """

    inds = _np.arange(len(self))
    if key == 'm' or key == 'mass': inds = _np.argsort(self.m)
    elif key == 'b' or key == 'impact' or key == 'impact param' or key == 'impact parameter': inds = _np.argsort(self.b)
    elif key == 'v' or key == 'vinf' or key == 'v_inf' or key == 'velocity': inds = _np.argsort(self.v)
    elif key == 'inc' or key == 'inclination' or key == 'i' or key == 'I': inds = _np.argsort(self.inc)
    elif key == 'omega' or key == 'ω' or key == 'argument_periastron': inds = _np.argsort(self.omega)
    elif key == 'Omega' or key == 'Ω' or key == 'longitude_ascending_node': inds = _np.argsort(self.Omega)
    elif key == 'q' or key == 'peri' or key == 'perihelion' or key == 'periastron' or key == 'periapsis': 
      if isinstance(sim, _rebound.Simulation):
        inds = _np.argsort(self.q(sim))
      else: raise InvalidParameterTypeException()
    elif key == 'e' or key == 'eccentricity': 
      if isinstance(sim, _rebound.Simulation):
        inds = _np.argsort(self.e(sim))
      else: raise InvalidParameterTypeException()
    elif _tools.isList(key): 
      print(key)
      if len(key) != len(self): raise ListLengthException(f'Difference of key: {len(key)} and stars: {len(self)}.')
      inds = _np.array(key)
    else: raise InvalidValueForKeyException()

    if argsort: return inds
    else:
      self.m[:] = self.m[inds]
      self.b[:] = self.b[inds]
      self.v[:] = self.v[inds]
      self.inc[:] = self.inc[inds]
      self.omega[:] = self.omega[inds]
      self.Omega[:] = self.Omega[inds]

  def save(self, filename):
    """
    Save the current instance of the Stars class to a file using pickle.

    Args:
      filename (str): The name of the file to save the instance to. The file will be saved in binary format.

    Example:
      ```python
      import airball
      se = airball.OpenCluster()
      stars = se.random_stars(100)
      stars.save('open_cluster.stars')
      ```
    """
    if not isinstance(filename, str): raise ValueError('Filename must be a string.')
    with open(filename, 'wb') as pfile:
      _pickle.dump(self, pfile, protocol=_pickle.HIGHEST_PROTOCOL)

  @classmethod
  def _load(cls, filename, init=False):
    """
    Load an instance of the Stars class from a file using pickle.

    Args:
      filename (str): The name of the file to load the instance from. The file should be in binary format, pickled.

    Returns:
      loaded_stars (Stars): The loaded instance of the Stars class.

    Example:
      ```python
      import airball
      stars = airball.Stars('open_cluster.stars')
      ```
    """
    if not isinstance(filename, str): raise ValueError('Filename must be a string.')
    return _pickle.load(open(filename, 'rb'))

  def stats(self, returned=False):
    '''
    Prints a summary of the current stats of the Stars object.
    The stats are returned as a string if `returned=True`.
    '''
    s = f"<{self.__module__}.{type(self).__name__} object at {hex(id(self))}, "
    s += f"N={f'{self.N:,.0f}' if len(self.shape) == 1 else self.shape}"
    if self.N > 0: s += f", m= {_np.min(self.m.value):,.2f}-{_np.max(self.m.value):,.2f} {self.units['mass']}"
    if self.N > 0: s += f", b= {_np.min(self.b.value):,.0f}-{_np.max(self.b.value):,.0f} {self.units['length']}"
    if self.N > 0: s += f", v= {_np.min(self.v.value):,.0f}-{_np.max(self.v.value):,.0f} {self.units['velocity']}"
    s += f"{f', Environment={self.environment.name}' if self.environment is not None else ''}"
    s += ">"
    if returned: return s
    else: print(s)
  
  def __str__(self):
    return self.stats(returned=True)
  
  def __repr__(self):
    return self.stats(returned=True)

  def __getitem__(self, key):
    int_types = int, _np.integer
  
    # Basic indexing.
    if isinstance(key, int_types):
      # If the set of Stars is multi-dimensional, return the requested subset of stars as a set of Stars.
      if len(self.m.shape) > 1: return Stars(m=self.m[key], b=self.b[key], v=self.v[key], inc=self.inc[key], omega=self.omega[key], Omega=self.Omega[key], UNIT_SYSTEM=self.units.UNIT_SYSTEM)
      # Otherwise return the requested Star.
      else: return Star(m=self.m[key], b=self.b[key], v=self.v[key], inc=self.inc[key], omega=self.omega[key], Omega=self.Omega[key], UNIT_SYSTEM=self.units.UNIT_SYSTEM)

    # Allows for boolean array masking and indexing using a subset of indices.
    if isinstance(key, _np.ndarray):
      return Stars(m=self.m[key], b=self.b[key], v=self.v[key], inc=self.inc[key], omega=self.omega[key], Omega=self.Omega[key], UNIT_SYSTEM=self.units.UNIT_SYSTEM)
    
    # Allow for speed efficient slicing by returning a new set of Stars which are a subset of the original object.
    if isinstance(key, slice):
      # Check for number of elements returned by the slice.
      numEl = _tools.numberOfElementsReturnedBySlice(*key.indices(self.N))
            # If the slice requests the entire set, then simply return the set.
            # if key == slice(None, None, None): return self #  !!! **Note: this is a reference to the same object.** !!!
      # If there are no elements requested, return the empty set.
      if numEl == 0: return Stars(m=[], b=[], v=[], size=0)
      # If only one element is requested, return a set of Stars with only one Star.
      elif numEl == 1: return Stars(m=self.m[key], b=self.b[key], v=self.v[key], inc=self.inc[key], omega=self.omega[key], Omega=self.Omega[key], UNIT_SYSTEM=self.units.UNIT_SYSTEM, size=1)
      # Otherwise return a subset of the Stars defined by the slice.
      else: return Stars(m=self.m[key], b=self.b[key], v=self.v[key], inc=self.inc[key], omega=self.omega[key], Omega=self.Omega[key], UNIT_SYSTEM=self.units.UNIT_SYSTEM)

    # Allow for Numpy style array indexing.
    if isinstance(key, tuple):
      # Check if Stars data is multi-dimensional.
      if len(self.m.shape) == 1: raise IndexError(f'Too many indices: Stars are 1-dimensional, but {len(key)} were indexed.')
      # Check to see if the tuple has a slice.
      hasSlice = _tools.hasTrue([isinstance(k, slice) for k in key])
      if hasSlice:
        # Check the number of elements requested by the slice.
        numEl = [_tools.numberOfElementsReturnedBySlice(*k.indices(self.m.shape[i])) if isinstance(k, slice) else 1 for i,k in enumerate(key)]
        # If there are no elements requested, return the empty set.
        if numEl.count(0) > 0: return Stars(m=[], b=[], v=[], size=0)
        # If multiple elements are requested, return a set of Stars.
        elif _np.any(_np.array(numEl) > 1): return Stars(m=self.m[key], b=self.b[key], v=self.v[key], inc=self.inc[key], omega=self.omega[key], Omega=self.Omega[key], UNIT_SYSTEM=self.units.UNIT_SYSTEM)
        # If only one element is requested, return a set of Stars with only one Star.
        else:
          # Check to see if the single element is an scalar or an array with only one element.
          if self.m[key].isscalar: return Stars(m=self.m[key], b=self.b[key], v=self.v[key], inc=self.inc[key], omega=self.omega[key], Omega=self.Omega[key], size=1, UNIT_SYSTEM=self.units.UNIT_SYSTEM)
          else: return Stars(m=self.m[key], b=self.b[key], v=self.v[key], inc=self.inc[key], omega=self.omega[key], Omega=self.Omega[key], UNIT_SYSTEM=self.units.UNIT_SYSTEM)
      # If there is no slice, the return the requested Star.
      else: return Star(m=self.m[key], b=self.b[key], v=self.v[key], inc=self.inc[key], omega=self.omega[key], Omega=self.Omega[key], UNIT_SYSTEM=self.units.UNIT_SYSTEM)

    raise _tools.InvalidKeyException() 

  def __setitem__(self, key, value):
    star_type = Star, Stars
    if isinstance(value, star_type):
      self.m[key], self.b[key], self.v[key], self.inc[key], self.omega[key], self.Omega[key] = value.params
    else: raise InvalidStarException()
  
  def __delitem__(self, key):
    raise ValueError('Cannot delete Star elements from Stars array.')

  def __iter__(self):
    for i in list(_np.ndindex(self.m.shape)):
      yield Star(m=self.m[i], b=self.b[i], v=self.v[i], inc=self.inc[i], omega=self.omega[i], Omega=self.Omega[i], UNIT_SYSTEM=self.units.UNIT_SYSTEM)

  def __len__(self):
    return self.shape[0]
  
  def __eq__(self, other):
    # Overrides the default implementation
    if isinstance(other, Stars):
        data = (_np.all(self.m == other.m) and _np.all(self.b == other.b) and _np.all(self.v == other.v) and _np.all(self.inc == other.inc) and _np.all(self.omega == other.omega) and _np.all(self.Omega == other.Omega))
        properties = (self.N == other.N and self.shape == self.shape and self.units == other.units and self.environment == other.environment)
        return data and properties
    return NotImplemented

  def __hash__(self):
    # Overrides the default implementation
    data = []
    for d in sorted(self.__dict__.items()):
        try: data.append((d[0], tuple(d[1])))
        except: data.append(d)
    data = tuple(data)
    return hash(data)
  
  def __add__(self, other):
    # Overrides the default implementation
    if isinstance(other, Stars):
      if self.N == 0 and other.N == 0: return Stars()
      else: return Stars(m=_np.concatenate((self.m, other.m)), b=_np.concatenate((self.b, other.b)), v=_np.concatenate((self.v, other.v)), inc=_np.concatenate((self.inc, other.inc)), omega=_np.concatenate((self.omega, other.omega)), Omega=_np.concatenate((self.Omega, other.Omega)))
    return NotImplemented

################################
###### Custom Exceptions #######
################################

class InvalidStarException(Exception):
  def __init__(self): super().__init__('Object is not a valid airball.Star object.')

class UnspecifiedParameterException(Exception):
  def __init__(self, message): super().__init__(message)

class OverspecifiedParametersException(Exception):
  def __init__(self, message): super().__init__(message)

class ListLengthException(Exception):
  def __init__(self, message): super().__init__(f'List arguments must be same length or shape. {message}')

class IncompatibleListException(Exception):
  def __init__(self): super().__init__('The given list type is not compatible. Please use a Python list or an ndarray.')

class InvalidValueForKeyException(Exception):
  def __init__(self, message=''): super().__init__(f'Invalid value for key. {message}')

class InvalidParameterTypeException(Exception):
  def __init__(self): super().__init__('The given parameter value is not a valid type.')

