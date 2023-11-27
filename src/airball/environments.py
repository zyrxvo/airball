import rebound as _rebound
import numpy as _numpy
from scipy.stats import uniform as _uniform
from scipy.stats import maxwell as _maxwell
from scipy.stats import expon as _exponential
from scipy.optimize import fminbound as _fminbound

from . import units as _u
from .imf import *
from .stars import *
from .analytic import *

class StellarEnvironment:
  '''
    This is the AIRBALL StellarEnvironment class.
    It encapsulates the relevant data for a static stellar environment.

    # Example
    my_env = airball.StellarEnvironment(stellar_density=10, velocity_dispersion=20, lower_mass_limit=0.08, upper_mass_limit=8, name='My Environment')
    my_star = my_env.random_star()

    If a `maximum_impact_parameter` is not given, AIRBALL attempts to estimate a reasonable one. 
    The Maximum Impact Parameter is radius defining the outer limit of the sphere of influence around a stellar system.
    There are predefined subclasses for the LocalNeighborhood, a generic OpenCluster, a generic GlobularCluster, and the Milky Way center GalacticBulge and GalacticCore.
  '''
  def __init__(self, stellar_density, velocity_dispersion, lower_mass_limit, upper_mass_limit, mass_function=None, maximum_impact_parameter=None, name=None, UNIT_SYSTEM=[], object_name=None, seed=None):

    # Check to see if an stars object unit is defined in the given UNIT_SYSTEM and if the user defined a different name for the objects.
    self.units = UnitSet(UNIT_SYSTEM)
    objectUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(_u.stars)]
    if objectUnit == [] and object_name is not None: self.units.object = _u.def_unit(object_name, _u.stars)
    elif objectUnit == [] and object_name is None: self.units.object = _u.stars
    else: self.units.object = objectUnit[0]

    self.density = stellar_density
    self.velocity_dispersion = velocity_dispersion

    self._upper_mass_limit = upper_mass_limit.to(self.units['mass']) if isQuantity(upper_mass_limit) else upper_mass_limit * self.units['mass']
    self._lower_mass_limit = lower_mass_limit.to(self.units['mass']) if isQuantity(lower_mass_limit) else lower_mass_limit * self.units['mass']
    self._IMF = IMF(min_mass=self._lower_mass_limit, max_mass=self._upper_mass_limit, mass_function=mass_function, unit=self.units['mass'])
    self._median_mass = self.IMF.median_mass
    self.maximum_impact_parameter = maximum_impact_parameter

    self.name = name if name is not None else 'Stellar Environment'
    self.seed = seed if seed is not None else None #_numpy.random.randint(0, int(2**32 - 1))

  def random_star(self, size=1, include_orientation=True, maximum_impact_parameter=None, **kwargs):
    ''' Alias for `random_stars`.'''
    return self.random_stars(size=size, include_orientation=include_orientation, maximum_impact_parameter=maximum_impact_parameter, **kwargs)

  def random_stars(self, size=1, include_orientation=True, maximum_impact_parameter=None, **kwargs):
    '''
      Computes a random star from a stellar environment.
      Returns: airball.Star() or airball.Stars() if size > 1.
    '''
    if isinstance(size, tuple): size = tuple([int(i) for i in size])
    else: size = int(size)

    self.seed = kwargs.get('seed')
    if self.seed != None: _numpy.random.seed(self.seed)

    v = _maxwell.rvs(scale=maxwell_boltzmann_scale_from_dispersion(self.velocity_dispersion), size=size) # Relative velocity of the star at infinity.

    max_impact = maximum_impact_parameter if maximum_impact_parameter is not None else self.maximum_impact_parameter
    b = max_impact * _numpy.sqrt(_uniform.rvs(size=size)) # Impact parameter of the star.

    m = self.IMF.random_mass(size=size) # Mass of the star.

    zeros = _numpy.zeros(size)
    inc = 2.0*_numpy.pi * _uniform.rvs(size=size) - _numpy.pi if include_orientation else zeros
    ω = 2.0*_numpy.pi * _uniform.rvs(size=size) - _numpy.pi if include_orientation else zeros
    Ω = 2.0*_numpy.pi * _uniform.rvs(size=size) - _numpy.pi if include_orientation else zeros

    if isinstance(size, tuple): return Stars(m=m, b=b, v=v, inc=inc, omega=ω, Omega=Ω, UNIT_SYSTEM=self.UNIT_SYSTEM)
    elif size > 1: return Stars(m=m, b=b, v=v, inc=inc, omega=ω, Omega=Ω, UNIT_SYSTEM=self.UNIT_SYSTEM)#, environment=self)
    else: return Star(m, b[0], v[0], inc[0], ω[0], Ω[0], UNIT_SYSTEM=self.UNIT_SYSTEM)

  def stats(self):
    '''
    Prints a summary of the current stats of the Stellar Environment.
    '''
    s = self.name
    s += "\n------------------------------------------\n"
    s += "{1} Density:     {0:12.4g} \n".format(self.density, "Stellar" if self.object_unit.to_string() == _u.stars.to_string() else "Object")
    s += "Velocity Scale:      {0:12.4g} \n".format(self.velocity_dispersion)
    s += "Mass Range:            {0:6.4g} - {1:1.4g}\n".format(self.lower_mass_limit.value, self.upper_mass_limit)
    s += "Median Mass:         {0:12.4g} \n".format(self.median_mass)
    s += "Max Impact Param:    {0:12.4g} \n".format(self.maximum_impact_parameter)
    s += "Encounter Rate:      {0:12.4g} \n".format(self.encounter_rate)
    s += "------------------------------------------"
    print(s)

  @property
  def object_unit(self):
    return self.units['object']
  
  @property
  def object_name(self):
    return self.units['object'].to_string()
  
  @object_name.setter
  def object_name(self, value):
    self.units.object = _u.def_unit(value, _u.stars)

  @property
  def UNIT_SYSTEM(self):
      return self.units.UNIT_SYSTEM

  @UNIT_SYSTEM.setter
  def UNIT_SYSTEM(self, UNIT_SYSTEM):
    self.units.UNIT_SYSTEM = UNIT_SYSTEM

  @property
  def median_mass(self):
    '''
      The median mass of the environment's IMF
    '''
    return self.IMF.median_mass.to(self.units['mass'])

  @property
  def maximum_impact_parameter(self):
    '''
      The largest impact parameter to affect a stellar system in the environment.
    '''
    return self._maximum_impact_parameter.to(self.units['length'])

  @maximum_impact_parameter.setter
  def maximum_impact_parameter(self, value):
    if value is not None:
      self._maximum_impact_parameter = value.to(self.units['length']) if isQuantity(value) else value * self.units['length']
    else:
      # TODO: Convert from fminbound to interpolation
      sim = _rebound.Simulation()
      sim.add(m=1.0)
      sim.add(m=5.2e-05, a=30.2, e=0.013)
      _f = lambda b: _numpy.log10(_numpy.abs(1e-16 - _numpy.abs(relative_energy_change(sim, Star(self.upper_mass_limit, b * self.units['length'], _numpy.sqrt(2.0)*maxwell_boltzmann_mean_from_dispersion(self.velocity_dispersion))))))
      bs = _numpy.logspace(1, 6, 1000)
      b0 = bs[_numpy.argmin(_f(bs))]
      self._maximum_impact_parameter = _fminbound(_f, b0/5, 5*b0) * self.units['length']

  @property
  def density(self):
    '''
      The number density of the environment.
      Default units: pc^{-3}.
    '''
    return self._density.to(self.units['density'])

  @density.setter
  def density(self, value):
    '''
      The number density of the environment.
      Default units: pc^{-3}.
    '''
    if isQuantity(value):
      if value.unit.is_equivalent(_u.stars/_u.m**3): self._density = value.to(self.units['density'])
      elif value.unit.is_equivalent(1/_u.m**3): self._density = (value * self.units['object']).to(self.units['density'])
      else: raise AssertionError('The given density units are not compatible.')
    else: self._density = value * self.units['density']

  @property
  def velocity_dispersion(self):
    '''
      Return the velocity dispersion of the environment.
      Default units: km/s.
    '''
    return self._velocity.to(self.units['velocity'])

  @velocity_dispersion.setter
  def velocity_dispersion(self, value):
    '''
      The velocity dispersion of the environment.
      Default units: km/s.
    '''
    self._velocity = value.to(self.units['velocity']) if isQuantity(value) else value * self.units['velocity']

  @property
  def velocity_mean(self):
    '''
      Return the velocity dispersion of the environment.
      Default units: km/s.
    '''
    return maxwell_boltzmann_mean_from_dispersion(self.velocity_dispersion).to(self.units['velocity'])

  @property
  def velocity_mode(self):
    '''
      Return the velocity dispersion of the environment.
      Default units: km/s.
    '''
    return maxwell_boltzmann_mode_from_dispersion(self.velocity_dispersion).to(self.units['velocity'])

  @property
  def velocity_rms(self):
    '''
      Return the velocity dispersion of the environment.
      Default units: km/s.
    '''

    v = _maxwell.rvs(scale=maxwell_boltzmann_scale_from_dispersion(self.velocity_dispersion), size=int(1e6))
    return verify_unit(_numpy.sqrt(_numpy.mean(v**2)), self.units['velocity'])

  @property
  def lower_mass_limit(self):
    '''
      Return the lower mass limit of the IMF of the environment.
      Default units: solMass
    '''
    return self.IMF.min_mass.to(self.units['mass'])

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
    return self.IMF.max_mass.to(self.units['mass'])

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
    if isinstance(value, IMF): self._IMF = IMF(value.min_mass, value.max_mass, value.imf, value.unit, value.number_samples, value.seed)
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
    return encounter_rate(self._density, maxwell_boltzmann_mean_from_dispersion(self.velocity_dispersion), self._maximum_impact_parameter, self.median_mass, unit_set=self.units).to(self.units['object']/self.units['time'])
  
  def cumulative_encounter_times(self, size=None):
    '''
        Returns the cumulative time from t=0 for when to the expect the next flyby encounters.
        Assumes a Poisson Process and uses an Exponential distribution with the encounter rate.
    '''
    if isinstance(size, tuple): 
      size = tuple([int(i) for i in size])
      result = _numpy.cumsum(_exponential.rvs(scale=1/self.encounter_rate, size=size), axis=1) << self.units['time']
      result -= result[:, 0][:, None]
      return result
    else:
      size = int(size)
      result = _numpy.cumsum(_exponential.rvs(scale=1/self.encounter_rate, size=size)) << self.units['time']
      result -= result[0]
      return result
  
  def encounter_times(self, size=None):
    '''
        Returns the time between encounters for when to the expect the next flyby encounters.
        Assumes a Poisson Process and uses an Exponential distribution with the encounter rate.
    '''
    if isinstance(size, tuple): 
      size = tuple([int(i) for i in size])
      return _exponential.rvs(scale=1/self.encounter_rate, size=size) << self.units['time']
    else:
      size = int(size)
      return _exponential.rvs(scale=1/self.encounter_rate, size=size) << self.units['time']

  def time_to_next_encounter(self):
    '''
        Draw a time to the next expected flyby encounter.
        Assumes a Poisson Process and uses an Exponential distribution with the encounter rate.
    '''
    return _exponential.rvs(scale=1/self.encounter_rate) * self.units['time']



class LocalNeighborhood(StellarEnvironment):
  '''
    This is a AIRBALL StellarEnvironment subclass for the Local Neighborhood.
    It encapsulates the relevant data for a static stellar environment representing the local neighborhood of the solar system.

    The stellar density is 0.14 pc^-3 defined by Bovy (2017).
    The velocity dispersion is 20 km/s, defined by Binnery & Tremaine (2008) v_rms ~50 km/s and Bailer-Jones et al. (2018) so that 90% of stars have v < 100 km/s with an encounter rate of ~20 stars/Myr within 1 pc.
    The mass limits is defined to between 0.08-8 solar masses using Chabrier (2003) for single stars when m < 1 and a power-law model from Bovy (2017) for stars m ≥ 1 to account for depleted stars due to stellar evolution.

    # Example
    my_local = airball.LocalNeighborhood()
    my_10stars = my_local.random_star(size=10)
    # returns a Stars object with the masses, impact parameters, velocities, and orientation of the 10 Star objects in a heliocentric model.
  '''
  short_name = 'Local'
  def local_mass_function(x):
    '''
      This defined using Chabrier (2003) for single stars when m < 1 and a power-law model from Bovy (2017) for stars m ≥ 1 to account for depleted stars due to stellar evolution.
    '''
    return chabrier_2003_single(1, 0.0567) * (x)**-4.7 if x > 1 else chabrier_2003_single(x, 0.0567)

  def __init__(self, stellar_density = 0.14 * _u.stars/_u.pc**3, velocity_dispersion = 20.8 * _u.km/_u.s, lower_mass_limit=0.08 * _u.solMass, upper_mass_limit = 8 * _u.solMass, maximum_impact_parameter=10000 * _u.au, UNIT_SYSTEM=[], mass_function=local_mass_function, object_name=None):
    super().__init__(stellar_density=stellar_density, velocity_dispersion=velocity_dispersion, lower_mass_limit=lower_mass_limit, upper_mass_limit=upper_mass_limit, mass_function=mass_function, maximum_impact_parameter=maximum_impact_parameter, UNIT_SYSTEM=UNIT_SYSTEM, name = 'Local Neighborhood', object_name=object_name)

class OpenCluster(StellarEnvironment):
  '''
    This is a AIRBALL StellarEnvironment subclass for a generic Open Cluster.
    It encapsulates the relevant data for a static stellar environment representing a generic open cluster.

    The stellar density is 100 pc^-3 informed by Adams (2010).
    The velocity scale is 1 km/s informed by Adams (2010) and Malmberg et al. (2011).
    The mass limit is defined to between 0.08-100 solar masses using Chabrier (2003) for single stars when m < 1 and Salpeter (1955) for stars m ≥ 1.

    # Example
    my_open = airball.OpenCluster()
    my_10stars = my_open.random_star(size=10)
    # returns a Stars object with the masses, impact parameters, velocities, and orientation of the 10 Star objects in a heliocentric model.
  '''
  short_name = 'Open'

  def __init__(self, stellar_density = 100 * _u.stars * _u.pc**-3, velocity_dispersion = 1 * _u.km/_u.s, lower_mass_limit=0.08 * _u.solMass, upper_mass_limit = 100 * _u.solMass, maximum_impact_parameter=1000 * _u.au, UNIT_SYSTEM=[], object_name=None):
    super().__init__(stellar_density=stellar_density, velocity_dispersion=velocity_dispersion, lower_mass_limit=lower_mass_limit, upper_mass_limit=upper_mass_limit, mass_function=None, maximum_impact_parameter=maximum_impact_parameter, UNIT_SYSTEM=UNIT_SYSTEM, name = 'Open Cluster', object_name=object_name)

class GlobularCluster(StellarEnvironment):
  short_name = 'Globular'

  def __init__(self, stellar_density = 1000 * _u.stars * _u.pc**-3, velocity_dispersion = 10 * _u.km/_u.s, lower_mass_limit=0.08 * _u.solMass, upper_mass_limit = 1 * _u.solMass, maximum_impact_parameter=5000 * _u.au, UNIT_SYSTEM=[], object_name=None):
    super().__init__(stellar_density=stellar_density, velocity_dispersion=velocity_dispersion, lower_mass_limit=lower_mass_limit, upper_mass_limit=upper_mass_limit, mass_function=None, maximum_impact_parameter=maximum_impact_parameter, UNIT_SYSTEM=UNIT_SYSTEM, name = 'Globular Cluster', object_name=object_name)

class GalacticBulge(StellarEnvironment):
  short_name = 'Bulge'

  def __init__(self, stellar_density = 50 * _u.stars * _u.pc**-3, velocity_dispersion = 120 * _u.km/_u.s, lower_mass_limit=0.08 * _u.solMass, upper_mass_limit = 10 * _u.solMass, maximum_impact_parameter=50000 * _u.au, UNIT_SYSTEM=[], object_name=None):
    super().__init__(stellar_density=stellar_density, velocity_dispersion=velocity_dispersion, lower_mass_limit=lower_mass_limit, upper_mass_limit=upper_mass_limit, mass_function=None, maximum_impact_parameter=maximum_impact_parameter, UNIT_SYSTEM=UNIT_SYSTEM, name = 'Milky Way Bulge', object_name=object_name)

class GalacticCore(StellarEnvironment):
  short_name = 'Core'

  def __init__(self, stellar_density = 10000 * _u.stars * _u.pc**-3, velocity_dispersion = 170 * _u.km/_u.s, lower_mass_limit=0.08 * _u.solMass, upper_mass_limit = 10 * _u.solMass, maximum_impact_parameter=50000 * _u.au, UNIT_SYSTEM=[_u.yr], object_name=None):
    super().__init__(stellar_density=stellar_density, velocity_dispersion=velocity_dispersion, lower_mass_limit=lower_mass_limit, upper_mass_limit=upper_mass_limit, mass_function=None, maximum_impact_parameter=maximum_impact_parameter, UNIT_SYSTEM=UNIT_SYSTEM, name = 'Milky Way Core', object_name=object_name)
