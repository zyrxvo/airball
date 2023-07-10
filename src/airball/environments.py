import rebound as _rebound
import numpy as _numpy
from scipy.stats import uniform as _uniform
from scipy.stats import maxwell as _maxwell
from scipy.optimize import fminbound as _fminbound

from . import units
from .imf import *
from .stars import *
from .analytic import *
from .tools import *

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

    v = _maxwell.rvs(scale=maxwell_boltzmann_scale_from_dispersion(self.velocity_dispersion), size=size) # Relative velocity of the star at infinity.

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
      _f = lambda b: _numpy.log10(_numpy.abs(1e-16 - _numpy.abs(relative_energy_change(sim, Star(self.upper_mass_limit, b, _numpy.sqrt(2.0)*maxwell_boltzmann_mean_from_dispersion(self.velocity_dispersion))))))
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
      else: raise AssertionError('The given density units are not compatible.')
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
  def velocity_mean(self):
    '''
      Return the velocity dispersion of the environment.
      Default units: km/s.
    '''
    return maxwell_boltzmann_mean_from_dispersion(self.velocity_dispersion).to(self.units.units['velocity'])
  
  @property
  def velocity_mode(self):
    '''
      Return the velocity dispersion of the environment.
      Default units: km/s.
    '''
    return maxwell_boltzmann_mode_from_dispersion(self.velocity_dispersion).to(self.units.units['velocity'])
  
  @property
  def velocity_rms(self):
    '''
      Return the velocity dispersion of the environment.
      Default units: km/s.
    '''

    v = _maxwell.rvs(scale=maxwell_boltzmann_scale_from_dispersion(self.velocity_dispersion), size=int(1e6))
    return verify_unit(_numpy.sqrt(_numpy.mean(v**2)), self.units.units['velocity'])

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
    return encounter_rate(self._density, maxwell_boltzmann_mean_from_dispersion(self.velocity_dispersion), self._maximum_impact_parameter, self.median_mass).to(self.units.units['object']/self.units.units['time'])



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

  def __init__(self, stellar_density = 0.14 * units.stars/units.pc**3, velocity_dispersion = 20.8 * units.km/units.s, lower_mass_limit=0.08 * units.solMass, upper_mass_limit = 8 * units.solMass, maximum_impact_parameter=10000 * units.au, UNIT_SYSTEM=[], mass_function=local_mass_function):
    super().__init__(stellar_density=stellar_density, velocity_dispersion=velocity_dispersion, lower_mass_limit=lower_mass_limit, upper_mass_limit=upper_mass_limit, mass_function=mass_function, maximum_impact_parameter=maximum_impact_parameter, UNIT_SYSTEM=UNIT_SYSTEM, name = 'Local Neighborhood')

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
  
  def __init__(self, stellar_density = 100 * units.stars * units.pc**-3, velocity_dispersion = 1 * units.km/units.s, lower_mass_limit=0.08 * units.solMass, upper_mass_limit = 100 * units.solMass, maximum_impact_parameter=1000 * units.au, UNIT_SYSTEM=[]):
    super().__init__(stellar_density=stellar_density, velocity_dispersion=velocity_dispersion, lower_mass_limit=lower_mass_limit, upper_mass_limit=upper_mass_limit, mass_function=None, maximum_impact_parameter=maximum_impact_parameter, UNIT_SYSTEM=UNIT_SYSTEM, name = 'Open Cluster')

class GlobularCluster(StellarEnvironment):
  short_name = 'Globular'
  
  def __init__(self, stellar_density = 1000 * units.stars * units.pc**-3, velocity_dispersion = 10 * units.km/units.s, lower_mass_limit=0.08 * units.solMass, upper_mass_limit = 1 * units.solMass, maximum_impact_parameter=5000 * units.au, UNIT_SYSTEM=[]):
    super().__init__(stellar_density=stellar_density, velocity_dispersion=velocity_dispersion, lower_mass_limit=lower_mass_limit, upper_mass_limit=upper_mass_limit, mass_function=None, maximum_impact_parameter=maximum_impact_parameter, UNIT_SYSTEM=UNIT_SYSTEM, name = 'Globular Cluster')

class GalacticBulge(StellarEnvironment):
  short_name = 'Bulge'
  
  def __init__(self, stellar_density = 50 * units.stars * units.pc**-3, velocity_dispersion = 120 * units.km/units.s, lower_mass_limit=0.08 * units.solMass, upper_mass_limit = 10 * units.solMass, maximum_impact_parameter=50000 * units.au, UNIT_SYSTEM=[]):
    super().__init__(stellar_density=stellar_density, velocity_dispersion=velocity_dispersion, lower_mass_limit=lower_mass_limit, upper_mass_limit=upper_mass_limit, mass_function=None, maximum_impact_parameter=maximum_impact_parameter, UNIT_SYSTEM=UNIT_SYSTEM, name = 'Milky Way Bulge')

class GalacticCore(StellarEnvironment):
  short_name = 'Core'
  
  def __init__(self, stellar_density = 10000 * units.stars * units.pc**-3, velocity_dispersion = 170 * units.km/units.s, lower_mass_limit=0.08 * units.solMass, upper_mass_limit = 10 * units.solMass, maximum_impact_parameter=50000 * units.au, UNIT_SYSTEM=[units.yr]):
    super().__init__(stellar_density=stellar_density, velocity_dispersion=velocity_dispersion, lower_mass_limit=lower_mass_limit, upper_mass_limit=upper_mass_limit, mass_function=None, maximum_impact_parameter=maximum_impact_parameter, UNIT_SYSTEM=UNIT_SYSTEM, name = 'Milky Way Core')

