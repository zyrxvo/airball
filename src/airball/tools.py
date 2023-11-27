import numpy as _numpy
import joblib as _joblib
import astropy.constants as const
from . import units as _u

twopi = 2.*_numpy.pi

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
        return _numpy.all([u1.is_equivalent(u2) for u1,u2 in zip(self, other)])
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
    if method == 'nan': ret = _numpy.nancumsum(a)
    elif method == 'nn':
        bool = _numpy.isnan(a)
        inds = _numpy.arange(len(a))[bool]
        ret = a.copy()
        for i in inds:
            ret[i] = (ret[i-1 if i-1 > 0 else i+1] + ret[i+1 if i+1 < len(a) else i-1])/2.0
        ret = _numpy.cumsum(ret)
    else: ret = _numpy.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# Implemented from StackOverflow: https://stackoverflow.com/a/33585850
def moving_median(arr, n=3):
    '''Compute the moving median of an array of numbers using the nearest n elements.'''
    idx = _numpy.arange(n) + _numpy.arange(len(arr)-n+1)[:,None]
    return _numpy.median(arr[idx], axis=1)

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
    geometric_means = lambda a: _numpy.sqrt(a[1:] * a[:-1])
    
    astart = _numpy.min(arr)
    aend = _numpy.max(arr)
    arange = _numpy.linspace(astart, aend, bins+1, endpoint=True)
    
    y,b = _numpy.histogram(arr, bins=arange, density=density)
    x = geometric_means(b)
    w = wfac * _numpy.mean(x[1:] - x[:-1])
    
    if normalize: return x, y/_numpy.trapz(y,x), w
    else: return x,y,w

def hist10(arr, bins=10, normalize=False, density=False, wfac=1):
    # https://stackoverflow.com/questions/30551694/logarithmic-multi-sequenz-plot-with-equal-bar-widths/30555229#30555229
    # """Return pairwise geometric means of adjacent elements."""
    geometric_means = lambda a: _numpy.sqrt(a[1:] * a[:-1])
    
    astart = _numpy.log10(_numpy.min(arr)/2)
    aend = _numpy.log10(_numpy.max(arr)*2)
    arange = _numpy.logspace(astart, aend, bins+1, endpoint=True)
    
    y,b = _numpy.histogram(arr, bins=arange, density=density)
    x = geometric_means(b)
    w = wfac * x*_numpy.mean((x[1:] - x[:-1])/x[:-1])
    
    if normalize: return x, y/_numpy.trapz(y,x), w
    else: return x,y,w

############################################################
############### Properties and Elements ####################
############################################################

def calculate_angular_momentum(sim):
    L = _numpy.zeros((sim.N, 3))
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
    return _numpy.sqrt(1 + (numerator/mu)**2.)

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
    return 1 + star_q * star_vinf * star_vinf / mu

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
    return verify_unit(star_q * _numpy.sqrt((star_e + 1.0)/(star_e - 1.0)), _u.au)

def gravitational_mu(sim, star):
    # Convert the units of the REBOUND Simulation into Astropy Units.
    units = rebound_units(sim)
    G = (sim.G * units['length']**3 / units['mass'] / units['time']**2)
    star_mass = verify_unit(star.mass, units['mass'])
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
    return star.b * _numpy.sqrt((star_e - 1.0)/(star_e + 1.0))

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
    a = -star.b/_numpy.sqrt(e**2. - 1.) # Compute the semi-major axis of the flyby star
    l = -a*(e*e-1.) # Compute the semi-latus rectum of the hyperbolic orbit to get the true anomaly (-a because the semi-major axis is negative)
    f = _numpy.arccos((l/rmax-1.)/e) # Compute the true anomaly

    return {'m':star.m.value, 'a':a.value, 'e':e.value, 'inc':star.inc.value, 'omega':star.omega.value, 'Omega':star.Omega.value, 'f':-f.value}, l.value

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
    return _numpy.sqrt((_numpy.pi*_numpy.square(sigma))/(3.0*_numpy.pi - 8.0))

def maxwell_boltzmann_scale_from_mean(mu):
    '''
        Converts mean (mu) to scale factor for Maxwell-Boltzmann distributions.
    '''
    return _numpy.sqrt(_numpy.pi/2.0) * (mu / 2.0)

def maxwell_boltzmann_mean_from_dispersion(sigma):
    '''
        Converts velocity dispersion (sigma) to mean (mu) for Maxwell-Boltzmann distributions.
    '''
    scale = maxwell_boltzmann_scale_from_dispersion(sigma)
    return (2.0 * scale) * _numpy.sqrt(2.0/_numpy.pi)

def maxwell_boltzmann_mode_from_dispersion(sigma):
    '''
        Converts velocity dispersion (sigma) to mode (most common or typical value) for Maxwell-Boltzmann distributions.
    '''
    scale = maxwell_boltzmann_scale_from_dispersion(sigma)
    return scale * _numpy.sqrt(2.0)

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

    return (_numpy.pi * R**2) * (1 + 2*G*(sun_mass + M)/(R * v**2))

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
    if isinstance(l,(list,_numpy.ndarray)): return True
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

