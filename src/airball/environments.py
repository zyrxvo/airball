import numpy as _numpy
from scipy.stats import uniform as _uniform
from scipy.stats import maxwell as _maxwell
from scipy.optimize import fminbound as _fminbound

from .flybys import *
from .analytic import *
from .tools import *


def _scale(sigma):
  '''
    Converts velocity dispersion (sigma) to scale factor for Maxwell-Boltzmann distributions.
  '''
  return _numpy.sqrt((_numpy.pi*_numpy.square(sigma))/(3.0*_numpy.pi - 8.0))

def _IMF(upper_limit, size=1):
  '''
    Returns an order or magnitude Initial Mass Function (IMF) generator for producing random mass samples.
  '''
  if upper_limit <= 1: return imf_gen_1(size)
  elif upper_limit <= 10: return imf_gen_10(size)
  elif upper_limit <= 100: return imf_gen_100(size)
  else: raise AssertionError('The upper mass limit is too high.')

class Star:
  '''
    This is the AIRBALL Star class.
    It encapsulates the relevant parameters for a given star.
    Only the mass is an quantity intrinsic to the object.
    The impact parameter, velocity, inclination, argument of periastron, and longitude of the ascending node quantities are defined with respect to the host star and plane passing through the star.
  '''
  def __init__(self, m, b, v, inc, omega, Omega, UNIT_SYSTEM=None) -> None:
      '''
      The Mass, m, (Msun), Impact Parameter, b, (AU), Velocity, v, (km/s)
      Inclination, inc, (rad), Argument of the Periastron, omega (rad), and Longitude of the Ascending Node, Omega, (rad)
      Or define an astropy.units system, i.e. UNIT_SYSTEM = [u.pc, u.Myr, u.solMass, u.rad].
      '''
      self._units = {'length': u.au, 'time': u.Myr, 'mass': u.solMass, 'angle': u.rad, 'velocity': u.km/u.s}
      if UNIT_SYSTEM is not None:
        self.UNIT_SYSTEM = UNIT_SYSTEM

      self._mass = m.to(self._units['mass']) if isQuantity(m) else m * self._units['mass']
      self._impact_parameter = b.to(self._units['length']) if isQuantity(b) else b * self._units['length']
      self._velocity = v.to(self._units['velocity']) if isQuantity(v) else v * self._units['velocity']
      self._inclination = inc.to(self._units['angle']) if isQuantity(inc) else inc * self._units['angle']
      self._argument_periastron = omega.to(self._units['angle']) if isQuantity(omega) else omega * self._units['angle']
      self._longitude_ascending_node = Omega.to(self._units['angle']) if isQuantity(Omega) else Omega * self._units['angle']

  @property
  def UNIT_SYSTEM(self):
      return self._UNIT_SYSTEM

  @UNIT_SYSTEM.setter
  def UNIT_SYSTEM(self, UNIT_SYSTEM):
    if UNIT_SYSTEM != []:
      lengthUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(u.m)]
      self._units['length'] = lengthUnit[0] if lengthUnit != [] else self._units['length']

      timeUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(u.s)]
      self._units['time'] = timeUnit[0] if timeUnit != [] else self._units['time']

      velocityUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(u.km/u.s)]
      if velocityUnit == [] and timeUnit != [] and lengthUnit != []: velocityUnit = [lengthUnit[0]/timeUnit[0]]
      self._units['velocity'] = velocityUnit[0] if velocityUnit != [] else self._units['velocity']

      unit = [this for this in UNIT_SYSTEM if this.is_equivalent(u.kg)]
      self._units['mass'] = unit[0] if unit != [] else self._units['mass']

      unit = [this for this in UNIT_SYSTEM if this.is_equivalent(u.rad)]
      self._units['angle'] = unit[0] if unit != [] else self._units['angle']
      self._UNIT_SYSTEM = list(self._units.values())

  @property
  def m(self):
      return self._mass.decompose(self.UNIT_SYSTEM)

  @m.setter
  def m(self, m):
      self._mass = m.to(u.solMass) if isQuantity(m) else m * self._units['mass']

  @property
  def mass(self):
      return self.m

  @mass.setter
  def mass(self, m):
      self.m = m
  
  @property
  def b(self):
      return self._impact_parameter.to(self._units['length'])

  @b.setter
  def b(self, b):
      self._impact_parameter = b.to(u.au) if isQuantity(b) else b * self._units['length']

  @property
  def impact_parameter(self):
      return self.b

  @impact_parameter.setter
  def impact_parameter(self, b):
      self.b = b

  @property
  def v(self):
      return self._velocity.to(self._units['velocity'])

  @v.setter
  def v(self, v):
      self._velocity = v.to(self._units['velocity']) if isQuantity(v) else v * self._units['velocity']

  @property
  def velocity(self):
      return self.v

  @velocity.setter
  def velocity(self, v):
      self.v = v

  @property
  def inc(self):
      return self._inclination.to(self._units['angle'])

  @inc.setter
  def inc(self, inc):
      self._inclination = inc.to(self._units['angle']) if isQuantity(inc) else inc * self._units['angle']

  @property
  def omega(self):
      return self._argument_periastron.decto(self._units['angle'])

  @omega.setter
  def omega(self, omega):
      self._argument_periastron = omega.to(self._units['angle']) if isQuantity(omega) else omega * self._units['angle']

  @property
  def Omega(self):
      return self._longitude_ascending_node.to(self._units['angle'])

  @Omega.setter
  def Omega(self, Omega):
      self._longitude_ascending_node = Omega.to(self._units['angle']) if isQuantity(Omega) else Omega * self._units['angle']

  def stats(self, returned=False):
    ''' 
    Prints a summary of the current stats of the Stellar Environment.
    '''
    s = f"<{self.__module__}.{type(self).__name__} object at {hex(id(self))}, "
    s += f"m= {self._mass.to(self._units['mass']):1.4g}, "
    s += f"b= {self._impact_parameter.to(self._units['length']):1.4g}, "
    s += f"v= {self._velocity.to(self._units['velocity']):1.4g}, "
    s += f"inc= {self._inclination.to(self._units['angle']):1.4g}, "
    s += f"omega= {self._argument_periastron.to(self._units['angle']):1.4g}, "
    s += f"Omega= {self._longitude_ascending_node.to(self._units['angle']):1.4g}>"
    if returned: return s
    else: print(s)
  
  def __str__(self):
     return self.stats(returned=True)
  
  def __repr__(self):
     return self.stats(returned=True)

class StarError(Exception):
    """Star was not found."""
    pass

class Stars():
  '''
    This class allows the user to access stars like a dictionary using the star's 1) index 2) hash 3) string (which will be converted to hash).
    Allows for negative indices and slicing.
  '''
  def __init__(self, starList) -> None:
     self._stars = [Star(*params) for params in starList]
     pass
   
  @property
  def stars(self):
     return self._stars
  
  @property
  def N(self):
     return len(self.stars)
  
  @property
  def median_mass(self):
    return _numpy.median([mass.value for mass in self.m]) * (self._stars[0].m).unit
  
  @property
  def mean_mass(self):
    return _numpy.mean([mass.value for mass in self.m]) * (self._stars[0].m).unit
  
  def __getitem__(self, key):
    int_types = int,
    
    if isinstance(key, slice):
        return [self[i] for i in range(*key.indices(len(self)))]

    if isinstance(key, int_types):
        if key < 0: # accept negative indices
            key += self.N
        if key < 0 or key >= self.N:
            raise AttributeError("Index {0} used to access particles out of range.".format(key))
        return self.stars[key]

    raise StarError("Invalid key for Stars.") 

  def __setitem__(self, key, value):
    if isinstance(value, Star):
        star = self[key]
        if star.index == -1:
            raise AttributeError("Can't set Star (Star not found).")
        else:
            self._stars[star.index] = value
  
  def __iter__(self):
    for star in self.stars:
        yield star

  def __len__(self):
    return len(self.stars)

  @property
  def m(self):
      return [star.mass for star in self.stars]

  @property
  def mass(self):
      return self.m
  
  @property
  def b(self):
      return [star.impact_parameter for star in self.stars]

  @property
  def impact_parameter(self):
      return self.b

  @property
  def v(self):
      return [star.velocity for star in self.stars]

  @property
  def velocity(self):
      return self.v

  @property
  def inc(self):
      return [star.inc for star in self.stars]

  @property
  def omega(self):
      return [star.omega for star in self.stars]

  @property
  def Omega(self):
      return [star.Omega for star in self.stars]


class StellarEnvironment:
  '''
    This is the AIRBALL StellarEnvironment class.
    It encapsulates the relevant data for a static stellar environment.

    # Example
    my_env = airball.StellarEnvironment(stellar_density=100, velocity_dispersion=10, upper_mass_limit=100, name='My Environment')
    my_star = my_env.random_star()

    If a `maximum_impact_parameter` is not given, AIRBALL attempts to estimate a reasonable one.
    There are predefined subclasses for the LocalNeighborhood, a generic OpenCluster, a generic GlobularCluster, and the Milky Way center GalacticBulge and GalacticCore.
  '''
  _maximum_impact_parameter = None
  def __init__(self, stellar_density, velocity_dispersion, upper_mass_limit, maximum_impact_parameter=None, name=None, UNIT_SYSTEM=None, object_name='stars'):
    self._UNIT_SYSTEM = [u.pc, u.Myr, u.solMass, u.rad] if UNIT_SYSTEM is None else UNIT_SYSTEM
    self._starUnit = u.def_unit(object_name)
    self._UNIT_SYSTEM.append(self.starUnit)

    self._density = stellar_density.decompose(self._UNIT_SYSTEM) if isQuantity(stellar_density) else (stellar_density * self.starUnit * u.pc**-3)
    self.velocity = velocity_dispersion.decompose(self._UNIT_SYSTEM) if isQuantity(velocity_dispersion) else velocity_dispersion * u.km / u.s

    self.mass_limit = upper_mass_limit.decompose(self._UNIT_SYSTEM) if isQuantity(upper_mass_limit) else upper_mass_limit * u.solMass
    assert self.mass_limit.value <= 100, 'The upper mass limit is too high.'
    self._median_mass = None
    if maximum_impact_parameter is not None:
      self._maximum_impact_parameter = maximum_impact_parameter.decompose(self._UNIT_SYSTEM) if isQuantity(maximum_impact_parameter) else maximum_impact_parameter * u.au

    self.name = name if name is not None else 'Stellar Environment'

  def random_star(self, maximum_impact_parameter=None, include_orientation=True, size=1):
    '''
      Computes a random star from a stellar environment.
      Returns: airball.Star() or airball.Stars() if size > 1.
    '''
    size = int(size)

    v = _maxwell.rvs(scale=_scale(self.velocity), size=size) # Velocity of the star at infinity.

    max_impact = maximum_impact_parameter if maximum_impact_parameter is not None else self.maximum_impact_parameter
    b = max_impact * _numpy.sqrt(_uniform.rvs(size=size)) # Impact parameter of the star.
    
    m = _IMF(self.mass_limit.value, size=size) # Mass of the star.
    
    zeros = _numpy.zeros(size)
    inc = 2.0*_numpy.pi * _uniform.rvs(size=size) - _numpy.pi if include_orientation else zeros
    ϖ = 2.0*_numpy.pi * _uniform.rvs(size=size) - _numpy.pi if include_orientation else zeros
    Ω = 2.0*_numpy.pi * _uniform.rvs(size=size) - _numpy.pi if include_orientation else zeros
    
    if size > 1: return Stars(_numpy.array([m,b,v,inc,ϖ,Ω]).T)
    else: return Star(m[0], b[0], v[0], inc[0], ϖ[0], Ω[0])

  def stats(self, UNIT_SYSTEM=None):
    ''' 
    Prints a summary of the current stats of the Stellar Environment.
    '''
    usys = []
    if UNIT_SYSTEM is None: usys = self._UNIT_SYSTEM 
    else:
      usys = UNIT_SYSTEM
      usys.append(self.starUnit)
    s = self.name
    s += "\n------------------------------------------\n"
    s += "Stellar Density:     {0:12.4g} \n".format(self.density.decompose(usys))
    s += "Velocity Scale:      {0:12.4g} \n".format(self.velocity.decompose(usys))
    s += "Mass Range:            {0:3.4g} - {1:3.4g}\n".format((0.01 * u.solMass).decompose(usys).value, self.mass_limit.decompose(usys))
    s += "Median Mass:         {0:12.4g} \n".format(self.median_mass.decompose(usys))
    s += "Max Impact Param:    {0:12.4g} \n".format(self.maximum_impact_parameter.decompose(usys))
    s += "Encounter Rate:      {0:12.4g} \n".format(self.encounter_rate().decompose(usys))
    s += "------------------------------------------"
    print(s)

  @property
  def starUnit(self):
    return self._starUnit

  @property
  def median_mass(self):
    '''
      Compute the rough median mass of the environment
    '''
    if self._median_mass is None: self._median_mass = _numpy.median(_IMF(self.mass_limit.value, size=int(1e6))) * u.solMass
    return self._median_mass
  
  @median_mass.setter
  def median_mass(self, value):
    self._median_mass = value.to(u.solMass) if isQuantity(value) else value * u.solMass

  @property
  def maximum_impact_parameter(self):
    '''
      Compute an estimate for the largest impact parameter to affect a Sun-Neptune system.
    '''
    if self._maximum_impact_parameter is None:
      _f = lambda b: _numpy.log10(_numpy.abs(1e-16 - _numpy.abs(relative_energy_change(1.0, 5.2e-05, 30.2, 0.013, self.mass_limit.value, b, _numpy.sqrt(2.0)*_scale(self.velocity.value)))))
      bs = _numpy.logspace(1, 6, 1000)
      b0 = bs[_numpy.argmin(_f(bs))]
      self._maximum_impact_parameter = _fminbound(_f, b0/5, 5*b0) * u.au
    return self._maximum_impact_parameter.to(u.au)
  
  @maximum_impact_parameter.setter
  def maximum_impact_parameter(self, value):
    self._maximum_impact_parameter = value.to(u.au) if isQuantity(value) else value * u.au

  @property
  def density(self):
    '''
      The number density of the environment in units of pc^{-3}.
    '''
    return self._density.decompose(self._UNIT_SYSTEM)

  @density.setter
  def density(self, value):
    '''
      The number density of the environment in units of pc^{-3}.
    '''
    self._density = value.decompose(self._UNIT_SYSTEM) if isQuantity(value) else value * self.starUnit * u.pc**-3

  @property
  def velocity_dispersion(self):
    '''
      Return the velocity dispersion of the environment in units of km/s.
    '''
    return self.velocity
  
  # @density.setter
  # def density(self, value):
  #   self._density = value * convert_pc3_to_au3

  def encounter_rate(self):
    '''
        Compute the expected flyby encounter rate Γ = ⟨nσv⟩ for the stellar environment in units of flybys per year.
        The inverse of the encouter rate will give the average number of years until a flyby.

        n : stellar number density in units of AU^{-3}
        σ : interaction cross section in units of AU^2
        v : velocity dispersion in units of km/s

        The interaction cross section σ = πb^2 considers gravitational focussing b = q√[1 + (2GM)/(q v∞^2)] and considers
        - the median mass of the environment
        - the maximum impact parameter
        - the relative velocity at infinity derived from the velocity dispersion
    '''
    return encounter_rate(self._density, self.velocity, self._maximum_impact_parameter, star_mass=self.median_mass).decompose(self._UNIT_SYSTEM)

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
    # returns a (3,10) numpy array with the masses, impact parameters, and velocities of the stars.
  '''
  _starUnit = u.def_unit('stars')
  _UNIT_SYSTEM = [u.pc, u.Myr, u.solMass, _starUnit]
  _maximum_impact_parameter = 10000 * u.au
  name = 'Local Neighborhood'
  short_name = 'Local'

  def __init__(self, maximum_impact_parameter=None):
    self._density = 0.14 * self.starUnit * u.pc**-3
    self.velocity = 26 * u.km/u.s
    self.mass_limit = 10 * u.solMass
    self._median_mass = None
    if maximum_impact_parameter is not None:
      self._maximum_impact_parameter = maximum_impact_parameter.to(u.au) if isQuantity(maximum_impact_parameter) else maximum_impact_parameter * u.au

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
    # returns a (3,10) numpy array with the masses, impact parameters, and velocities of the stars.
  '''
  _starUnit = u.def_unit('stars')
  _UNIT_SYSTEM = [u.pc, u.Myr, u.solMass, _starUnit]
  _maximum_impact_parameter = 1000 * u.au
  name = 'Open Cluster'
  short_name = 'Open'
  
  def __init__(self):
    self._density = 100 * self.starUnit * u.pc**-3
    self.velocity = 1 * u.km/u.s
    self.mass_limit = 100 * u.solMass
    self._median_mass = None

class GlobularCluster(StellarEnvironment):
  _starUnit = u.def_unit('stars')
  _UNIT_SYSTEM = [u.pc, u.Myr, u.solMass, _starUnit]
  _maximum_impact_parameter = 5000 * u.au
  name = 'Globular Cluster'
  short_name = 'Globular'
  
  def __init__(self):
    self._density = 1000 * self.starUnit * u.pc**-3
    self.velocity = 10 * u.km/u.s
    self.mass_limit = 1 * u.solMass
    self._median_mass = None

class GalacticBulge(StellarEnvironment):
  _starUnit = u.def_unit('stars')
  _UNIT_SYSTEM = [u.pc, u.Myr, u.solMass, _starUnit]
  _maximum_impact_parameter = 50000 * u.au
  name = 'Milky Way Bulge'
  short_name = 'Bulge'
  
  def __init__(self):
    self._density = 50 * self.starUnit * u.pc**-3
    self.velocity = 120 * u.km/u.s
    self.mass_limit = 10 * u.solMass
    self._median_mass = None

class GalacticCore(StellarEnvironment):
  _starUnit = u.def_unit('stars')
  _UNIT_SYSTEM = [u.pc, u.Myr, u.solMass, _starUnit]
  _maximum_impact_parameter = 50000 * u.au
  name = 'Milky Way Core'
  short_name = 'Core'
  
  def __init__(self):
    self._density = 10000 * self.starUnit * u.pc**-3
    self.velocity = 170 * u.km/u.s
    self.mass_limit = 10 * u.solMass
    self._median_mass = None