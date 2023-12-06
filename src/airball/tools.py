import numpy as _np
import joblib as _joblib
import astropy.constants as const
from . import units as _u
from scipy.stats import uniform
from scipy.stats import maxwell
from scipy.stats import expon

twopi = 2.*_np.pi

class UnitSet():

  def __init__(self, UNIT_SYSTEM=[]) -> None:
    self._units = {'length': _u.au, 'time': _u.Myr, 'mass': _u.solMass, 'angle': _u.rad, 'velocity': _u.km/_u.s, 'object': _u.stars, 'density': _u.stars/_u.pc**3}
    self.UNIT_SYSTEM = UNIT_SYSTEM
    pass

  @property
  def units(self):
    return self._units

  @property
  def UNIT_SYSTEM(self):
    return self._UNIT_SYSTEM
  
  def __getitem__(self, key):
    if isinstance(key, str): return self.units[key]
    else: raise InvalidKeyException() 
  
  def __setitem__(self, key, value):
    if isinstance(key, str):
      if isUnit(value): self.units[key] = value
      else: raise InvalidUnitException() 
    else: raise InvalidKeyException() 

  def __str__(self):
    s = '{'
    for key in self.units:
       s += f'{key}: {self.units[key].to_string()}, '
    s = s[:-2] + '}'
    return s
  
  def __repr__(self):
    s = '{'
    for key in self.units:
       s += f'{key}: {self.units[key].to_string()},\n'
    s = s[:-2] + '}'
    return s
  
  def __iter__(self):
    for k in self.units:
      yield self.units[k]
  
  def __eq__(self, other):
    '''Determines if two UnitSets are equivalent, not necessarily identical.'''
    if isinstance(other, UnitSet):
        return _np.all([u1.is_equivalent(u2) for u1,u2 in zip(self, other)])
    return NotImplemented
  
  def __ne__(self, other):
    """Overrides the default implementation (unnecessary in Python 3)"""
    x = self.__eq__(other)
    if x is not NotImplemented:
        return not x
    return NotImplemented

  def __hash__(self):
    """Overrides the default implementation"""
    return hash(tuple(sorted(self.__dict__.items())))
  

  @property
  def length(self):
    """Units of LENGTH."""
    return self._units['length']

  @length.setter
  def length(self, value):
    """
    Setter for the Units of LENGTH.

    Parameters:
    - value: An Astropy Quantity describing LENGTH.
    """
    self.UNIT_SYSTEM = [value]

  @property
  def time(self):
    """Units of TIME."""
    return self._units['time']

  @time.setter
  def time(self, value):
    """
    Setter for the Units of TIME.

    Parameters:
    - value: An Astropy Quantity describing TIME.
    """
    self.UNIT_SYSTEM = [value]

  @property
  def mass(self):
    """Units of MASS."""
    return self._units['mass']

  @mass.setter
  def mass(self, value):
    """
    Setter for the Units of MASS.

    Parameters:
    - value: An Astropy Quantity describing MASS.
    """
    self.UNIT_SYSTEM = [value]

  @property
  def angle(self):
    """Units of ANGLE."""
    return self._units['angle']

  @angle.setter
  def angle(self, value):
    """
    Setter for the Units of ANGLE.

    Parameters:
    - value: An Astropy Quantity describing ANGLE.
    """
    self.UNIT_SYSTEM = [value]

  @property
  def velocity(self):
    """Units of VELOCITY."""
    return self._units['velocity']

  @velocity.setter
  def velocity(self, value):
    """
    Setter for the Units of VELOCITY.

    Parameters:
    - value: An Astropy Quantity describing VELOCITY.
    """
    self.UNIT_SYSTEM = [value]

  @property
  def density(self):
    """Units of DENSITY."""
    return self._units['density']

  @density.setter
  def density(self, value):
    """
    Setter for the Units of DENSITY .

    Parameters:
    - value: An Astropy Quantity describing DENSITY.
    """
    self.UNIT_SYSTEM = [value]

  @property
  def object(self):
    """Units of an OBJECT (such as a star)."""
    return self._units['object']

  @object.setter
  def object(self, value):
    """
    Setter for the Units of an OBJECT (such as a star).

    Parameters:
    - value: An Astropy Quantity describing an OBJECT (such as a star).
    """
    self.UNIT_SYSTEM = [value]

  @UNIT_SYSTEM.setter
  def UNIT_SYSTEM(self, UNIT_SYSTEM):
    if UNIT_SYSTEM != []:
      lengthUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(_u.m)]
      self._units['length'] = lengthUnit[0] if lengthUnit != [] else self._units['length']

      timeUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(_u.s)]
      self._units['time'] = timeUnit[0] if timeUnit != [] else self._units['time']

      velocityUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(_u.km/_u.s)]
      if velocityUnit == [] and timeUnit != [] and lengthUnit != []: velocityUnit = [lengthUnit[0]/timeUnit[0]]
      self._units['velocity'] = velocityUnit[0] if velocityUnit != [] else self._units['velocity']

      massUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(_u.kg)]
      self._units['mass'] = massUnit[0] if massUnit != [] else self._units['mass']

      angleUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(_u.rad)]
      self._units['angle'] = angleUnit[0] if angleUnit != [] else self._units['angle']

      objectUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(_u.stars)]
      self._units['object'] = objectUnit[0] if objectUnit != [] else _u.stars

      densityUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(_u.stars/_u.m**3)]
      densityUnit2 = [this for this in UNIT_SYSTEM if this.is_equivalent(1/_u.m**3)]
      if densityUnit == [] and densityUnit2 != []: 
        densityUnit = [self._units['object'] * densityUnit2[0]]
      elif densityUnit == [] and objectUnit != [] and lengthUnit != []: 
        densityUnit = [self._units['object']/self._units['length']**3]
      elif densityUnit == [] and densityUnit2 == [] and objectUnit != []:
         densityLength = [this for this in self._units['density'].bases if this.is_equivalent(_u.m)][0]
         densityUnit = [self._units['object']/densityLength**3]
      self._units['density'] = densityUnit[0] if densityUnit != [] else self._units['density']
    
    self._UNIT_SYSTEM = list(self._units.values())



############################################################
################### Helper Functions #######################
############################################################

# Implemented from StackOverflow: https://stackoverflow.com/a/14314054
def moving_average(a, n=3, method=None) :
    '''Compute the moving average of an array of numbers using the nearest n elements.'''
    if method == 'nan': ret = _np.nancumsum(a)
    elif method == 'nn':
        bool = _np.isnan(a)
        inds = _np.arange(len(a))[bool]
        ret = a.copy()
        for i in inds:
            ret[i] = (ret[i-1 if i-1 > 0 else i+1] + ret[i+1 if i+1 < len(a) else i-1])/2.0
        ret = _np.cumsum(ret)
    else: ret = _np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# Implemented from StackOverflow: https://stackoverflow.com/a/33585850
def moving_median(arr, n=3):
    '''Compute the moving median of an array of numbers using the nearest n elements.'''
    idx = _np.arange(n) + _np.arange(len(arr)-n+1)[:,None]
    return _np.median(arr[idx], axis=1)

def save_as_simulationarchive(filename, sims, deletefile=True):
    '''
    Saves a list of REBOUND Simulations as a SimulationArchive.
    '''
    for i,s in enumerate(sims):
        s.simulationarchive_snapshot(filename, deletefile=(deletefile if i == 0 else False))

# From REBOUND particle.py
def notNone(a):
    """
    Returns True if array a contains at least one element that is not None. Returns False otherwise.
    """
    return a.count(None) != len(a)

def hasTrue(a):
    """
    Returns True if array a contains at least one element that is True. Returns False otherwise.
    """
    return a.count(True) > 0

def numberOfElementsReturnedBySlice(start, stop, step):
   return (stop - start) // step

def _integrate(sim, tmax):
    sim.integrate(tmax)
    return sim

def integrate(sims, tmaxs, n_jobs=-1, verbose=0):
    sim_results = _joblib.Parallel(n_jobs=n_jobs, verbose=verbose, require='sharedmem')(
        _joblib.delayed(_integrate)(sim=sims[int(i)], tmax=tmaxs[int(i)]) for i in range(len(sims)))
    return sim_results

def hist(arr, bins=10, normalize=False, density=False, wfac=1):
    # https://stackoverflow.com/questions/30551694/logarithmic-multi-sequenz-plot-with-equal-bar-widths/30555229#30555229
    # """Return pairwise geometric means of adjacent elements."""
    geometric_means = lambda a: _np.sqrt(a[1:] * a[:-1])
    
    astart = _np.min(arr)
    aend = _np.max(arr)
    arange = _np.linspace(astart, aend, bins+1, endpoint=True)
    
    y,b = _np.histogram(arr, bins=arange, density=density)
    x = geometric_means(b)
    w = wfac * _np.mean(x[1:] - x[:-1])
    
    if normalize: return x, y/_np.trapz(y,x), w
    else: return x,y,w

def hist10(arr, bins=10, normalize=False, density=False, wfac=1):
    # https://stackoverflow.com/questions/30551694/logarithmic-multi-sequenz-plot-with-equal-bar-widths/30555229#30555229
    # """Return pairwise geometric means of adjacent elements."""
    geometric_means = lambda a: _np.sqrt(a[1:] * a[:-1])
    
    astart = _np.log10(_np.min(arr)/2)
    aend = _np.log10(_np.max(arr)*2)
    arange = _np.logspace(astart, aend, bins+1, endpoint=True)
    
    y,b = _np.histogram(arr, bins=arange, density=density)
    x = geometric_means(b)
    w = wfac * x*_np.mean((x[1:] - x[:-1])/x[:-1])
    
    if normalize: return x, y/_np.trapz(y,x), w
    else: return x,y,w

# https://stackoverflow.com/a/13849249/71522
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / _np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return _np.arccos(_np.clip(_np.dot(v1_u, v2_u), -1.0, 1.0))

def reb_mod2pi(f):
    return _np.mod(twopi + _np.mod(f, twopi), twopi)

def reb_M_to_E(e, M):
  E = 0
  if e < 1.0 :
    M = reb_mod2pi(M); # avoid numerical artefacts for negative numbers
    E = M if e < 0.8 else _np.pi
    F = E - e*_np.sin(E) - M
    for i in range(100):
      E = E - F/(1.-e*_np.cos(E))
      F = E - e*_np.sin(E) - M
      if _np.all(_np.abs(F) < 1.0e-16) : break
    E = reb_mod2pi(E)
    return E
  else:
    E = M/_np.abs(M)*_np.log(2.*_np.abs(M)/e + 1.8)
    F = E - e*_np.sinh(E) + M
    for i in range(100):
      E = E - F/(1.0 - e*_np.cosh(E))
      F = E - e*_np.sinh(E) + M
      if _np.all(_np.abs(F) < 1.0e-16): break
    return E

def reb_E_to_f(e, E):
	if e > 1. :return reb_mod2pi(2.*_np.arctan(_np.sqrt((1.+e)/(e-1.))*_np.tanh(0.5*E)));
	else: return reb_mod2pi(2.*_np.arctan(_np.sqrt((1.+e)/(1.-e))*_np.tan(0.5*E)));


def reb_M_to_f(e, M):
  E = reb_M_to_E(e, M)
  return reb_E_to_f(e, E)


############################################################
############### Properties and Elements ####################
############################################################

def calculate_angular_momentum(sim):
    L = _np.zeros((sim.N, 3))
    L[0] = sim.angular_momentum()
    for i,p in enumerate(sim.particles[1:]):
        L[i+1,0] = p.m*(p.y*p.vz - p.z*p.vy)
        L[i+1,1] = p.m*(p.z*p.vx - p.x*p.vz)
        L[i+1,2] = p.m*(p.x*p.vy - p.y*p.vx)
    return L

def vinf_and_b_to_e(mu, star_b, star_v):
    '''
        Using the impact parameter to convert from the relative velocity at infinity between the two stars to the eccentricity of the flyby star.
        Equation (2) from Spurzem et al. (2009) https://ui.adsabs.harvard.edu/abs/2009ApJ...697..458S/abstract

        Parameters
        ----------
        mu : the total mass of the system (Sun, planets, and flyby star) times the gravitational constant G
        star_b :  impact parameter of the flyby star
        star_v : the relative velocity at infinity between the central star and the flyby star (hyperbolic excess velocity)
    '''

    star_b = verify_unit(star_b, _u.au)
    star_v = verify_unit(star_v, _u.km/_u.s)

    numerator = star_b * star_v**2.
    return _np.sqrt(1 + (numerator/mu)**2.) * _u.dimensionless_unscaled

def vinf_and_q_to_e(mu, star_q, star_v):
    '''
        Using the perihelion to convert from the relative velocity at infinity between the two stars to the eccentricity of the flyby star.

        Parameters
        ----------
        mu : the total mass of the system (Sun, planets, and flyby star) times the gravitational constant G
        star_q :  perihelion of the flyby star
        star_v : the relative velocity at infinity between the central star and the flyby star (hyperbolic excess velocity)
    '''

    star_q = verify_unit(star_q, _u.au)
    star_vinf = verify_unit(star_v, _u.km/_u.s)
    return (1 + star_q * star_vinf * star_vinf / mu) * _u.dimensionless_unscaled

def vinf_and_q_to_b(mu, star_q, star_v):
    '''
        Using the perihelion to convert from the relative velocity at infinity between the two stars to the eccentricity of the flyby star.

        Parameters
        ----------
        mu : the total mass of the system (Sun, planets, and flyby star) times the gravitational constant G
        star_q :  perihelion of the flyby star
        star_v : the relative velocity at infinity between the central star and the flyby star (hyperbolic excess velocity)
    '''

    mu = verify_unit(mu, (_u.au**3)/(_u.yr2pi**2))
    star_q = verify_unit(star_q, _u.au)
    star_vinf = verify_unit(star_v, _u.km/_u.s)
    star_e = 1 + star_q * star_vinf * star_vinf / mu
    return verify_unit(star_q * _np.sqrt((star_e + 1.0)/(star_e - 1.0)), _u.au)

def gravitational_mu(sim, star=None, star_mass=None):
    # Convert the units of the REBOUND Simulation into Astropy Units.
    units = rebound_units(sim)
    G = (sim.G * units['length']**3 / units['mass'] / units['time']**2)
    if star is not None and star_mass is not None: raise Exception('Cannot define both star and star_mass.')
    elif star is not None and star_mass is None: star_mass = verify_unit(star.mass, units['mass'])
    elif star is None and star_mass is not None: star_mass = verify_unit(star_mass, units['mass'])
    else: raise Exception('Either star or star_mass must be defined.')
    return G * (system_mass(sim)  * units['mass'] + star_mass)

def star_q(sim, star):
    '''
        Using the impact parameter and the relative velocity at infinity between the two stars convert to the perhelion of the flyby star.

        Parameters
        ----------
        mu : the total mass of the system (Sun, planets, and flyby star) times the gravitational constant G
        star_q :  perihelion of the flyby star
        star_v : the relative velocity at infinity between the central star and the flyby star (hyperbolic excess velocity)
    '''

    units = rebound_units(sim)
    G = (sim.G * units['length']**3 / units['mass'] / units['time']**2)
    mu = G * (system_mass(sim)  * units['mass'] + star.m)

    star_e = vinf_and_b_to_e(mu, star.b, star.v)
    return star.b * _np.sqrt((star_e - 1.0)/(star_e + 1.0))

def system_mass(sim):
    '''
        The total bound mass of the system.
    '''
    total_mass = 0
    for i,p in enumerate(sim.particles):
        if i == 0: total_mass += p.m
        elif p.a > 0: total_mass += p.m
        else: pass
    return total_mass      

def determine_eccentricity(sim, star_mass, star_b, star_v):
    '''Calculate the eccentricity of the flyby star. '''
    # Convert the units of the REBOUND Simulation into Astropy Units.
    units = rebound_units(sim)
    G = (sim.G * units['length']**3 / units['mass'] / units['time']**2)
    star_mass = verify_unit(star_mass, units['mass'])
    star_b = verify_unit(star_b, units['length'])
    star_v = verify_unit(star_v, units['length']/units['time'])

    mu = G * (system_mass(sim) * units['mass'] + star_mass)
    return vinf_and_b_to_e(mu=mu, star_b=star_b, star_v=star_v)

def initial_conditions_from_stellar_params(sim, star, rmax):
    '''
        Calculate the flyby star's initial conditions based on the provided Simulation and starting distance (rmax).
    '''
    e = determine_eccentricity(sim, star.m, star.b, star.v)
    a = -star.b/_np.sqrt(e**2. - 1.) # Compute the semi-major axis of the flyby star
    l = -a*(e*e-1.) # Compute the semi-latus rectum of the hyperbolic orbit to get the true anomaly (-a because the semi-major axis is negative)
    if rmax == 0 * _u.au: f = 0 * _u.dimensionless_unscaled
    else: f = _np.arccos((l/rmax-1.)/e) # Compute the true anomaly

    return {'m':star.m.value, 'a':a.value, 'e':e.value, 'inc':star.inc.value, 'omega':star.omega.value, 'Omega':star.Omega.value, 'f':-f.value}, l.value

def hyperbolic_elements_from_stellar_params(sim, star, rmax):
    '''
        Calculate the flyby star's hyperbolic orbital elements based on the provided Simulation and starting distance (rmax).
    '''
    sim_units = rebound_units(sim)
    e = determine_eccentricity(sim, star.m, star.b, star.v)
    a = -star.b/_np.sqrt(e**2. - 1.) # Compute the semi-major axis of the flyby star
    l = -a*(e*e-1.) # Compute the semi-latus rectum of the hyperbolic orbit to get the true anomaly (-a because the semi-major axis is negative)
    if rmax == 0 * _u.au: f = 0 * _u.dimensionless_unscaled
    else: f = _np.arccos((l/rmax-1.)/e) # Compute the true anomaly

    G = (sim.G * sim_units['length']**3 / sim_units['mass'] / sim_units['time']**2)
    mu = G * (system_mass(sim) * sim_units['mass'] + star.m)

    # Compute the time to periapsis from the switching point (-a because the semi-major axis is negative).
    with _u.set_enabled_equivalencies(_u.dimensionless_angles()):
        E = _np.arccosh((_np.cos(f)+e)/(1.+e*_np.cos(f))) # Compute the eccentric anomaly
        M = e * _np.sinh(E)-E # Compute the mean anomaly
    Tperi = M/_np.sqrt(mu/(-a*a*a))

    return {'m':star.m, 'a':a, 'e':e, 'inc':star.inc, 'omega':star.omega, 'Omega':star.Omega, 'f':-f, 'T':Tperi, 'l':l}

def impulse_gradient(star):
    '''Calculate the impulse gradient for a flyby star.'''
    G = (1 * _u.au**3 / _u.solMass / _u.yr2pi**2)
    return ((2.0 * G * star.m) / (star.v * star.b**2.0)).to(_u.km/_u.s/_u.au)

############################################################
############# Stellar Environment Functions ################
############################################################


def maxwell_boltzmann_scale_from_dispersion(sigma):
    '''
        Converts velocity dispersion (sigma) to scale factor for Maxwell-Boltzmann distributions.
    '''
    return _np.sqrt((_np.pi*_np.square(sigma))/(3.0*_np.pi - 8.0))

def maxwell_boltzmann_scale_from_mean(mu):
    '''
        Converts mean (mu) to scale factor for Maxwell-Boltzmann distributions.
    '''
    return _np.sqrt(_np.pi/2.0) * (mu / 2.0)

def maxwell_boltzmann_mean_from_dispersion(sigma):
    '''
        Converts velocity dispersion (sigma) to mean (mu) for Maxwell-Boltzmann distributions.
    '''
    scale = maxwell_boltzmann_scale_from_dispersion(sigma)
    return (2.0 * scale) * _np.sqrt(2.0/_np.pi)

def maxwell_boltzmann_mode_from_dispersion(sigma):
    '''
        Converts velocity dispersion (sigma) to mode (most common or typical value) for Maxwell-Boltzmann distributions.
    '''
    scale = maxwell_boltzmann_scale_from_dispersion(sigma)
    return scale * _np.sqrt(2.0)

def cross_section(M, R, v, unit_set=UnitSet()):
    '''
        The cross-section with gravitational focusing.
        
        Parameters
        ----------
        M : the mass of flyby star (default units: solMass)
        R : the maximum interaction radius (default units: AU)
        v : the typical velocity from the distribution (default units: km/s)
    '''

    # Newton's gravitational constant in units of Msun, AU, and Years/2pi (G ~ 1).
    G = const.G.decompose(unit_set.UNIT_SYSTEM) 
    sun_mass = 1 * _u.solMass # mass of the Sun in units of Msun

    v = verify_unit(v, unit_set.units['velocity'])
    R = verify_unit(R, unit_set.units['length'])
    M = verify_unit(M, unit_set.units['mass'])

    return (_np.pi * R**2) * (1 + 2*G*(sun_mass + M)/(R * v**2))

def encounter_rate(n, v, R, M=(1 * _u.solMass), unit_set=UnitSet()):
    '''
        The expected flyby encounter rate within an stellar environment
        
        Parameters
        ----------
        n : stellar number density (default units: pc^{-3})
        v : average velocity  (default units: km/s)
        R : interaction radius (default units: AU)
        M : mass of a typical flyby star (default units: solMass)
    '''
    n = verify_unit(n, unit_set.units['density'])
    v = verify_unit(v, unit_set.units['velocity'])
    R = verify_unit(R, unit_set.units['length'])
    M = verify_unit(M, unit_set.units['mass'])

    return n * v * cross_section(M, R, v, unit_set)


############################################################
################### Units Functions ########################
############################################################

def rebound_units(sim):
    defrebunits = {'length': _u.au, 'mass': _u.solMass, 'time': _u.yr2pi}
    simunits = sim.units
    
    for unit in simunits:
        if simunits[unit] == None: simunits[unit] = defrebunits[unit]
        else: simunits[unit] = _u.Unit(simunits[unit])
    return simunits

def verify_unit(value, unit):
    return value.to(unit) if isQuantity(value) else value * unit

def isList(l):
    '''Determines if an object is a list or numpy array. Used for flyby parallelization.'''
    if isinstance(l,(list,_np.ndarray)): return True
    else: return False

def isQuantity(var):
    '''Determines if an object is an Astropy Quantity. Used for Stellar Environment initializations.'''
    return isinstance(var, _u.quantity.Quantity)

def isUnit(var):
    '''Determines if an object is an Astropy Quantity. Used for Stellar Environment initializations.'''
    return isinstance(var, (_u.core.IrreducibleUnit, _u.core.CompositeUnit, _u.Unit))


############################################################
###################### Exceptions ##########################
############################################################

class InvalidKeyException(Exception):
  def __init__(self): super().__init__('Invalid key type.')

class InvalidUnitException(Exception):
  def __init__(self): super().__init__('Value is not a valid unit type.')

