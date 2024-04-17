import rebound as _rebound
import numpy as _np
from scipy.stats import uniform as _uniform
from scipy.stats import maxwell as _maxwell
from scipy.stats import expon as _exponential
from scipy.optimize import fminbound as _fminbound

from . import units as _u
from . import imf as _imf
from .imf import IMF as _IMF
from .stars import Star as _Star
from .stars import Stars as _Stars
from . import analytic as _analytic
from . import tools as _tools

class StellarEnvironment:
  '''
    This is the AIRBALL StellarEnvironment class.
    It encapsulates the relevant data for a static stellar environment.

    Initializing a StellarEnvironment instance.

    Args:
      stellar_density (float): The stellar density of the environment.
      velocity_dispersion (float): The velocity dispersion in the environment.
      lower_mass_limit (float): The lower mass limit for stars in the environment.
      upper_mass_limit (float): The upper mass limit for stars in the environment.
      mass_function (callable, optional): A function that defines the mass distribution of stars. Default is None.
      maximum_impact_parameter (float, optional): The maximum impact parameter defining the outer limit of the sphere of influence around a stellar system. If not provided, AIRBALL attempts to estimate a reasonable one. Default is None.
      name (str, optional): The name of the environment. Default is None.
      UNIT_SYSTEM (list, optional): The unit system used in the environment. Default is an empty list.
      units (airball.units.UnitSet, optional): The units used in the environment. Default is None.
      object_name (str, optional): The name of the object in the environment. Default is None.
      seed (int, optional): The seed fixing the random star generator. Default is None so it's always random.

    Example:
      ```python
      import airball
      my_env = airball.StellarEnvironment(stellar_density=10, velocity_dispersion=20, lower_mass_limit=0.08, upper_mass_limit=8, name='My Environment')
      my_star = my_env.random_star()
      ```

    If a `maximum_impact_parameter` is not given, AIRBALL attempts to estimate a reasonable one.
    The Maximum Impact Parameter is radius defining the outer limit of the sphere of influence around a stellar system.
    There are predefined subclasses for the LocalNeighborhood, a generic OpenCluster, a generic GlobularCluster, and the Milky Way center GalacticBulge and GalacticCore.
  '''
  def __init__(self, stellar_density, velocity_dispersion, lower_mass_limit, upper_mass_limit, mass_function=None, maximum_impact_parameter=None, name=None, UNIT_SYSTEM=[], units=None, object_name=None, seed=None, number_imf_samples=100):
    # Check to see if an stars object unit is defined in the given UNIT_SYSTEM and if the user defined a different name for the objects.
    self.units = units if isinstance(units, _u.UnitSet) else _u.UnitSet(UNIT_SYSTEM)
    objectUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(_u.stars)]
    if objectUnit == [] and object_name is not None: self.units.object = _u.def_unit(object_name, _u.stars)
    elif objectUnit == [] and object_name is None: self.units.object = _u.stars
    else: self.units.object = objectUnit[0]

    self.density = stellar_density
    self.velocity_dispersion = velocity_dispersion

    self._upper_mass_limit = upper_mass_limit.to(self.units['mass']) if _tools.isQuantity(upper_mass_limit) else upper_mass_limit * self.units['mass']
    self._lower_mass_limit = lower_mass_limit.to(self.units['mass']) if _tools.isQuantity(lower_mass_limit) else lower_mass_limit * self.units['mass']
    self._IMF = _IMF(min_mass=self._lower_mass_limit, max_mass=self._upper_mass_limit, mass_function=mass_function, unit=self.units['mass'], number_samples=number_imf_samples, seed=seed)
    self._median_mass = self.IMF.median_mass
    self.maximum_impact_parameter = maximum_impact_parameter

    self.name = name if name is not None else 'Stellar Environment'
    self.short_name = self.name.replace(' ', '')
    self.seed = seed if seed is not None else None #_np.random.randint(0, int(2**32 - 1))

  def random_stars(self, size=1, **kwargs):
    '''
      Computes a random star from a stellar environment.

      Args:
        size (int or tuple): The number of stars to generate. If size is a tuple, it is interpreted as array dimensions. Default: 1.

      Keyword Args:
        include_orientation (bool, optional): If True, the orientation of the star is randomly generated. Otherwise, the orientation of the stars are zero. Default: True.
        maximum_impact_parameter (float, optional): The maximum impact parameter of the star. If None, the maximum impact parameter is estimated. Default: None.
        seed (int, optional): The random seed to use. If None is given then it is random every time. Default: None.

      Returns:
        stars (Star or Stars): A Star object or Stars object (if size > 1) with the randomly generated masses, impact parameters, velocities, and orientations in a heliocentric model.

      Example:
        ```python
        import airball
        my_env = airball.StellarEnvironment(stellar_density=10, velocity_dispersion=20, lower_mass_limit=0.08, upper_mass_limit=8, name='My Environment')
        my_stars = my_env.random_stars(10)
        ```
    '''
    if isinstance(size, tuple): size = tuple([int(i) for i in size])
    else: size = int(size)

    include_orientation = kwargs.get('include_orientation', True)
    maximum_impact_parameter = kwargs.get('maximum_impact_parameter')
    seed = kwargs.get('seed')

    self.seed = seed
    if self.seed != None: _np.random.seed(self.seed)

    v = _maxwell.rvs(scale=_tools.maxwell_boltzmann_scale_from_dispersion(self.velocity_dispersion), size=size) << self.units['velocity'] # Relative velocity of the star at infinity.

    max_impact = maximum_impact_parameter if maximum_impact_parameter is not None else self.maximum_impact_parameter
    b = max_impact * _np.sqrt(_uniform.rvs(size=size)) # Impact parameter of the star.

    m = self.IMF.random_mass(size=size) # Mass of the star.

    zeros = _np.zeros(size)
    inc = (2.0*_np.pi * _uniform.rvs(size=size) - _np.pi) << self.units['angle'] if include_orientation else zeros
    ω = (2.0*_np.pi * _uniform.rvs(size=size) - _np.pi) << self.units['angle'] if include_orientation else zeros
    Ω = (2.0*_np.pi * _uniform.rvs(size=size) - _np.pi) << self.units['angle'] if include_orientation else zeros

    if isinstance(size, tuple): return _Stars(m=m, b=b, v=v, inc=inc, omega=ω, Omega=Ω, UNIT_SYSTEM=self.UNIT_SYSTEM, environment=self)
    elif size > 1: return _Stars(m=m, b=b, v=v, inc=inc, omega=ω, Omega=Ω, UNIT_SYSTEM=self.UNIT_SYSTEM, environment=self)
    else: return _Star(m, b[0], v[0], inc[0], ω[0], Ω[0], UNIT_SYSTEM=self.UNIT_SYSTEM)

  def random_star(self, size=1, **kwargs):
    # Alias for `random_stars`
    return self.random_stars(size=size, **kwargs)

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

  def copy(self):
    '''
    Returns a deep copy of the current Stellar Environment.
    '''
    kwargs = {'stellar_density':self.density, 'velocity_dispersion':self.velocity_dispersion, 'lower_mass_limit':self.lower_mass_limit, 'upper_mass_limit':self.upper_mass_limit, 'mass_function':self.IMF.initial_mass_function, 'maximum_impact_parameter':self.maximum_impact_parameter, 'name':self.name, 'UNIT_SYSTEM':self.UNIT_SYSTEM, 'object_name':self.object_name, 'seed':self.seed, 'number_imf_samples':self.IMF.number_samples}
    return type(self)(**kwargs)

  def __eq__(self, other):
    # Overrides the default implementation
    if isinstance(other, StellarEnvironment):
        return (self.density == other.density and self.velocity_dispersion == other.velocity_dispersion and self.lower_mass_limit == other.lower_mass_limit and self.upper_mass_limit == other.upper_mass_limit and self.IMF == other.IMF and self.maximum_impact_parameter == other.maximum_impact_parameter and self.name == self.name and self.units == other.units and self.object_name == other.object_name and self.seed == other.seed)
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
    s += f", n= {self.density:1.4g}"
    s += f", v= {self.velocity_dispersion:,.1f}"
    s += f", m= {self.lower_mass_limit.value:,.2f}-{self.upper_mass_limit.value:,.1f} {self.units['mass']}"
    s += ">"
    if returned: return s
    else: print(s)

  def __str__(self):
    return self.summary(returned=True)

  def __repr__(self):
    return self.summary(returned=True)


  @property
  def object_unit(self):
    '''The unit of the object (star) in the environment.'''
    return self.units['object']

  @property
  def object_name(self):
    '''
    Args:
      value (str): The name of the object (star) in the environment.
    '''
    return self.units['object'].to_string()

  @object_name.setter
  def object_name(self, value):
    self.units.object = _u.def_unit(value, _u.stars)

  @property
  def UNIT_SYSTEM(self):
      '''
      Args:
        value (list of Units): A list of the units to use for the environment.
      '''
      return self.units.UNIT_SYSTEM

  @UNIT_SYSTEM.setter
  def UNIT_SYSTEM(self, UNIT_SYSTEM):
    self.units.UNIT_SYSTEM = UNIT_SYSTEM

  @property
  def median_mass(self):
    '''
      The median mass of the environment's initial mass function (IMF).
    '''
    return self.IMF.median_mass.to(self.units['mass'])

  @property
  def maximum_impact_parameter(self):
    '''
      The largest impact parameter to affect a stellar system in the environment. See the examples in [Adiabatic Tests](../examples/adiabatic-tests.ipynb/#stellarenvironmentmaximum_impact_parameter) for more details.
    '''
    return self._maximum_impact_parameter.to(self.units['length'])

  @maximum_impact_parameter.setter
  def maximum_impact_parameter(self, value):
    if value is not None:
      self._maximum_impact_parameter = value.to(self.units['length']) if _tools.isQuantity(value) else value * self.units['length']
    else:
      # TODO: Convert from fminbound to interpolation
      sim = _rebound.Simulation()
      sim.add(m=1.0)
      sim.add(m=5.2e-05, a=30.2, e=0.013) # Use Neptune as a test planet.
      _f = lambda b: _np.log10(_np.abs(1e-16 - _np.abs(_analytic.relative_energy_change(sim, _Star(self.upper_mass_limit, b * self.units['length'], _np.sqrt(2.0)*_tools.maxwell_boltzmann_mean_from_dispersion(self.velocity_dispersion))))))
      bs = _np.logspace(1, 6, 1000)
      b0 = bs[_np.argmin(_f(bs))]
      self._maximum_impact_parameter = _fminbound(_f, b0/5, 5*b0) * self.units['length']

  @property
  def density(self):
    '''
      Args:
        value (Quantity): The number density of the environment. Default units: $\\rm{pc}^{-3}$.
    '''
    return self._density.to(self.units['density'])

  @density.setter
  def density(self, value):
    if _tools.isQuantity(value):
      if value.unit.is_equivalent(_u.stars/_u.m**3): self._density = value.to(self.units['density'])
      elif value.unit.is_equivalent(1/_u.m**3): self._density = (value * self.units['object']).to(self.units['density'])
      else: raise AssertionError('The given density units are not compatible.')
    else: self._density = value * self.units['density']

  @property
  def velocity_dispersion(self):
    '''
      Args:
        value (Quantity): the velocity dispersion of the environment. Default units: km/s.
    '''
    return self._velocity.to(self.units['velocity'])

  @velocity_dispersion.setter
  def velocity_dispersion(self, value):
    self._velocity = value.to(self.units['velocity']) if _tools.isQuantity(value) else value * self.units['velocity']

  @property
  def velocity_mean(self):
    '''The mean velocity of the environment. Default units: km/s.'''
    return _tools.maxwell_boltzmann_mean_from_dispersion(self.velocity_dispersion).to(self.units['velocity'])

  @property
  def velocity_mode(self):
    '''Return the most common velocity of the environment. Default units: km/s.'''
    return _tools.maxwell_boltzmann_mode_from_dispersion(self.velocity_dispersion).to(self.units['velocity'])

  @property
  def velocity_rms(self):
    '''Return the root-mean-square velocity of the environment. Default units: km/s.'''
    v = _maxwell.rvs(scale=_tools.maxwell_boltzmann_scale_from_dispersion(self.velocity_dispersion), size=int(1e6))
    return _tools.verify_unit(_np.sqrt(_np.mean(v**2)), self.units['velocity'])

  @property
  def lower_mass_limit(self):
    '''
    Args:
      value (Quantity): The lower mass limit of the initial mass function (IMF) of the environment. Default units: $M_\\odot$.
    '''
    return self.IMF.min_mass.to(self.units['mass'])

  @lower_mass_limit.setter
  def lower_mass_limit(self, value):
    self.IMF.min_mass = value

  @property
  def upper_mass_limit(self):
    '''
    Args:
      value (Quantity): The upper mass limit of the initial mass function (IMF) of the environment. Default units: $M_\\odot$.
    '''
    return self.IMF.max_mass.to(self.units['mass'])

  @upper_mass_limit.setter
  def upper_mass_limit(self, value):
    self.IMF.max_mass = value

  @property
  def IMF(self):
    '''
      Args:
        value (IMF): The initial mass function (IMF) of the environment. An `airball.IMF` object.
    '''
    return self._IMF

  @IMF.setter
  def IMF(self, value):
    if isinstance(value, _IMF): self._IMF = _IMF(value.min_mass, value.max_mass, value.imf, value.unit, value.number_samples, value.seed)
    else: raise AssertionError('Initial Mass Function (IMF) must be an airball.IMF object.')

  @property
  def encounter_rate(self):
    '''
        Compute the expected flyby encounter rate $\\Gamma = ⟨nσv⟩$ for the stellar environment in units of flybys per year.
        The inverse of the encounter rate will give the average number of years until a flyby.

        The encounter rate is computed using the following parameters:

        - n : stellar number density. Default units: $\\rm{pc}^{-3}$
        - σ : interaction cross section. Default units: $\\rm{AU}^2$
        - v : velocity dispersion. Default units: $\\rm{km/s}$

        The interaction cross section $σ = πb^2$ considers gravitational focussing where $b = q \\sqrt(1 + \\frac{2GM}{q v_∞^2})$ determined by the median mass of the environment, the maximum impact parameter, and the relative velocity at infinity derived from the velocity dispersion.
    '''
    return _tools.encounter_rate(self._density, self.velocity_mean, self._maximum_impact_parameter, self.median_mass, unit_set=self.units).to(self.units['object']/self.units['time'])

  def cumulative_encounter_times(self, size):
    '''
    Returns the cumulative time from t=0 for when to expect the next flyby encounters.
    This function assumes a Poisson Process and uses an Exponential distribution with the encounter rate.

    Args:
        size (int or tuple): The shape of the returned array. If size is an integer, it is treated as the length of the array. If size is a tuple, it is treated as the shape of the array.

    Returns:
        times (Quantity): An array of cumulative encounter times. The shape of the array is determined by the size parameter.

    Example:
        ```python
        import airball
        my_env = airball.StellarEnvironment(stellar_density=10, velocity_dispersion=20, lower_mass_limit=0.08, upper_mass_limit=8, name='My Environment')
        my_env.cumulative_encounter_times(10) # returns an array of 10 cumulative encounter times.
        ```
    '''
    if isinstance(size, tuple):
      size = tuple([int(i) for i in size])
      result = _np.cumsum(_exponential.rvs(scale=1/self.encounter_rate, size=size), axis=1) << self.units['time']
      result -= result[:, 0][:, None]
      return result
    else:
      size = int(size)
      result = _np.cumsum(_exponential.rvs(scale=1/self.encounter_rate, size=size)) << self.units['time']
      result -= result[0]
      return result

  def encounter_times(self, size):
    '''
    Returns the time between encounters for when to the expect the next flyby encounters.
    Assumes a Poisson Process and uses an Exponential distribution with the encounter rate.

    Args:
        size (int or tuple): The shape of the returned array. If size is an integer, it is treated as the length of the array. If size is a tuple, it is treated as the shape of the array.

    Returns:
        times (Quantity): An array of encounter times. The shape of the array is determined by the size parameter.

    Example:
        ```python
        import airball
        my_env = airball.StellarEnvironment(stellar_density=10, velocity_dispersion=20, lower_mass_limit=0.08, upper_mass_limit=8, name='My Environment')
        my_env.encounter_times(10) # returns an array of 10 encounter times.
        ```
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

    Returns:
        times (Quantity): The next encounter time.

    Example:
        ```python
        import airball
        my_env = airball.StellarEnvironment(stellar_density=10, velocity_dispersion=20, lower_mass_limit=0.08, upper_mass_limit=8, name='My Environment')
        my_env.time_to_next_encounter()
        ```
    '''
    return _exponential.rvs(scale=1/self.encounter_rate) * self.units['time']



class LocalNeighborhood(StellarEnvironment):
  '''
    This is a `StellarEnvironment` subclass for the Local Neighborhood.
    It encapsulates the relevant data for a static stellar environment representing the local neighborhood of the solar system.

    The stellar density is 0.14 $\\rm{pc}^{-3}$ defined by [Bovy (2017)](https://ui.adsabs.harvard.edu/abs/2017MNRAS.470.1360B/abstract).
    The velocity distribution is defined using a Maxwell-Boltzmann distribution where the velocity dispersion is 20 km/s, defined by [Binnery & Tremaine (2008)](https://ui.adsabs.harvard.edu/abs/2008gady.book.....B/abstract) where the $v_\\rm{rms} \\sim 50$ km/s and [Bailer-Jones et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018A%26A...616A..37B/abstract) so that 90% of stars have v < 100 km/s with an encounter rate of ~20 stars/Myr within 1 pc. However, a more accurate representation of the velocity distribution in the solar neighborhood is a triaxial Gaussian distribution, but that has not been implemented here.
    The mass limits is defined to between 0.08-8 solar masses using Equation (17) from [Chabrier (2003)](https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C/abstract) for single stars when m < 1 and a power-law model from [Bovy (2017)](https://ui.adsabs.harvard.edu/abs/2017MNRAS.470.1360B/abstract) for stars m ≥ 1 to account for depleted stars due to stellar evolution.

    Example:
      ```python
      import airball
      my_local = airball.LocalNeighborhood()
      my_10stars = my_local.random_stars(size=10) # returns a Stars object with the masses, impact parameters, velocities, and orientation of the 10 Star objects in a heliocentric model.
      ```
  '''
  short_name = 'Local'
  def local_mass_function(x):
    '''
      This defined using Equation (17) from [Chabrier (2003)](https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C/abstract) for single stars when $m < 1$ and a power-law model from [Bovy (2017)](https://ui.adsabs.harvard.edu/abs/2017MNRAS.470.1360B/abstract) for stars $m \\ge 1$ to account for depleted stars due to stellar evolution.
    '''
    chabrier03 = _imf.chabrier_2003_single(1) #0.0567
    power_law = _imf.power_law(-4.7, chabrier03(1))
    return _np.where(x > 1, power_law(x), chabrier03(x))

  def __init__(self, stellar_density = 0.14 * _u.stars/_u.pc**3, velocity_dispersion = 20.8 * _u.km/_u.s, lower_mass_limit=0.08 * _u.solMass, upper_mass_limit = 8 * _u.solMass, mass_function=local_mass_function, maximum_impact_parameter=10000 * _u.au, UNIT_SYSTEM=[], units=None, name = 'Local Neighborhood', object_name=None, seed=None, number_imf_samples=100):
    super().__init__(stellar_density=stellar_density, velocity_dispersion=velocity_dispersion, lower_mass_limit=lower_mass_limit, upper_mass_limit=upper_mass_limit, mass_function=mass_function, maximum_impact_parameter=maximum_impact_parameter, UNIT_SYSTEM=UNIT_SYSTEM, units=units, name=name, object_name=object_name, seed=seed,number_imf_samples=number_imf_samples)

class OpenCluster(StellarEnvironment):
  '''
    This is a StellarEnvironment subclass for a generic Open Cluster.
    It encapsulates the relevant data for a static stellar environment representing a generic open cluster.

    The stellar density is 100 $\\rm{pc}^{-3}$ informed by [Adams (2010)](https://ui.adsabs.harvard.edu/abs/2010ARA%26A..48...47A/abstract).
    The velocity scale is 1 km/s informed by [Adams (2010)](https://ui.adsabs.harvard.edu/abs/2010ARA%26A..48...47A/abstract) and [Malmberg, Davies, & Heggie (2011)](https://ui.adsabs.harvard.edu/abs/2011MNRAS.411..859M/abstract).
    The mass limit is defined to between 0.08-100 solar masses using Equation (17) from [Chabrier (2003)](https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C/abstract) for single stars when m < 1 and [Salpeter (1955)](https://ui.adsabs.harvard.edu/abs/1955ApJ...121..161S/abstract) for stars m ≥ 1.

    Example:
      ```python
      import airball
      my_open = airball.OpenCluster()
      my_10stars = my_open.random_stars(size=10) # returns a Stars object with the masses, impact parameters, velocities, and orientation of the 10 Star objects in a heliocentric model.
      ```
  '''
  short_name = 'Open'

  def __init__(self, stellar_density = 100 * _u.stars * _u.pc**-3, velocity_dispersion = 1 * _u.km/_u.s, lower_mass_limit=0.08 * _u.solMass, upper_mass_limit = 100 * _u.solMass, mass_function=None, maximum_impact_parameter=1000 * _u.au, UNIT_SYSTEM=[], units=None, name = 'Open Cluster', object_name=None, seed=None, number_imf_samples=100):
    super().__init__(stellar_density=stellar_density, velocity_dispersion=velocity_dispersion, lower_mass_limit=lower_mass_limit, upper_mass_limit=upper_mass_limit, mass_function=mass_function, maximum_impact_parameter=maximum_impact_parameter, UNIT_SYSTEM=UNIT_SYSTEM, units=units, name=name, object_name=object_name, seed=seed,number_imf_samples=number_imf_samples)

class GlobularCluster(StellarEnvironment):
  '''
    This is a StellarEnvironment subclass for a generic Globular Cluster.
    It encapsulates the relevant data for a static stellar environment representing a generic globular cluster.

    The stellar density is 1000 $\\rm{pc}^{-3}$.
    The velocity scale is 10 km/s.
    The mass limit is defined to between 0.08-1 solar masses using Equation (17) from [Chabrier (2003)](https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C/abstract) for single stars when m < 1. It is assumed that there are no stellar masses greater than 1 solar mass in a globular cluster due to stellar evolution.

    Example:
      ```python
      import airball
      my_glob = airball.GlobularCluster()
      my_10stars = my_glob.random_stars(size=10) # returns a Stars object with the masses, impact parameters, velocities, and orientation of the 10 Star objects in a heliocentric model.
      ```
  '''
  short_name = 'Globular'

  def __init__(self, stellar_density = 1000 * _u.stars * _u.pc**-3, velocity_dispersion = 10 * _u.km/_u.s, lower_mass_limit=0.08 * _u.solMass, upper_mass_limit = 1 * _u.solMass, mass_function=None, maximum_impact_parameter=5000 * _u.au, UNIT_SYSTEM=[], units=None, name = 'Globular Cluster', object_name=None, seed=None, number_imf_samples=100):
    super().__init__(stellar_density=stellar_density, velocity_dispersion=velocity_dispersion, lower_mass_limit=lower_mass_limit, upper_mass_limit=upper_mass_limit, mass_function=mass_function, maximum_impact_parameter=maximum_impact_parameter, UNIT_SYSTEM=UNIT_SYSTEM, units=units, name=name, object_name=object_name, seed=seed,number_imf_samples=number_imf_samples)

class GalacticBulge(StellarEnvironment):
  '''
    This is a StellarEnvironment subclass for a generic Galactic Bulge.
    It encapsulates the relevant data for a static stellar environment representing a generic galactic bulge. This region of the galaxy is more dense than the typical field stars found in spiral arms and has a higher velocity dispersion.

    The stellar density is 50 $\\rm{pc}^{-3}$.
    The velocity scale is 120 km/s.
    The mass limit is defined to between 0.08-10 solar masses using Equation (17) from [Chabrier (2003)](https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C/abstract) for single stars when m < 1 and [Salpeter (1955)](https://ui.adsabs.harvard.edu/abs/1955ApJ...121..161S/abstract) for stars m ≥ 1.

    Example:
      ```python
      import airball
      my_bulge = airball.GalacticBulge()
      my_10stars = my_bulge.random_stars(size=10) # returns a Stars object with the masses, impact parameters, velocities, and orientation of the 10 Star objects in a heliocentric model.
      ```
  '''
  short_name = 'Bulge'

  def __init__(self, stellar_density = 50 * _u.stars * _u.pc**-3, velocity_dispersion = 120 * _u.km/_u.s, lower_mass_limit=0.08 * _u.solMass, upper_mass_limit = 10 * _u.solMass, mass_function=None, maximum_impact_parameter=50000 * _u.au, UNIT_SYSTEM=[], units=None, name = 'Milky Way Bulge', object_name=None, seed=None, number_imf_samples=100):
    super().__init__(stellar_density=stellar_density, velocity_dispersion=velocity_dispersion, lower_mass_limit=lower_mass_limit, upper_mass_limit=upper_mass_limit, mass_function=mass_function, maximum_impact_parameter=maximum_impact_parameter, UNIT_SYSTEM=UNIT_SYSTEM, units=units, name=name, object_name=object_name, seed=seed,number_imf_samples=number_imf_samples)

class GalacticCore(StellarEnvironment):
  '''
    This is a StellarEnvironment subclass for a generic Galactic Core.
    It encapsulates the relevant data for a static stellar environment representing a generic galactic core. This is the densest region of the galaxy and has the highest velocity dispersion.

    The stellar density is $10^4$ $\\rm{pc}^{-3}$.
    The velocity scale is 170 km/s.
    The mass limit is defined to between 0.08-10 solar masses using Equation (17) from [Chabrier (2003)](https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C/abstract) for single stars when m < 1 and [Salpeter (1955)](https://ui.adsabs.harvard.edu/abs/1955ApJ...121..161S/abstract) for stars m ≥ 1.

    Example:
      ```python
      import airball
      my_core = airball.GalacticCore()
      my_10stars = my_core.random_stars(size=10) # returns a Stars object with the masses, impact parameters, velocities, and orientation of the 10 Star objects in a heliocentric model.
      ```
  '''
  short_name = 'Core'

  def __init__(self, stellar_density = 10000 * _u.stars * _u.pc**-3, velocity_dispersion = 170 * _u.km/_u.s, lower_mass_limit=0.08 * _u.solMass, upper_mass_limit = 10 * _u.solMass, mass_function=None, maximum_impact_parameter=50000 * _u.au, UNIT_SYSTEM=[_u.yr], units=None, name = 'Milky Way Core', object_name=None, seed=None, number_imf_samples=100):
    super().__init__(stellar_density=stellar_density, velocity_dispersion=velocity_dispersion, lower_mass_limit=lower_mass_limit, upper_mass_limit=upper_mass_limit, mass_function=mass_function, maximum_impact_parameter=maximum_impact_parameter, UNIT_SYSTEM=UNIT_SYSTEM, units=units, name=name, object_name=object_name, seed=seed,number_imf_samples=number_imf_samples)
