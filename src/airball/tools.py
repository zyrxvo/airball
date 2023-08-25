import numpy as _numpy
import joblib as _joblib
import astropy.constants as const
from . import units as u

twopi = 2.*_numpy.pi

class UnitSet():

  def __init__(self, UNIT_SYSTEM=[]) -> None:
    self._units = {'length': u.au, 'time': u.Myr, 'mass': u.solMass, 'angle': u.rad, 'velocity': u.km/u.s, 'object': u.stars, 'density': u.stars/u.pc**3}
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

      massUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(u.kg)]
      self._units['mass'] = massUnit[0] if massUnit != [] else self._units['mass']

      angleUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(u.rad)]
      self._units['angle'] = angleUnit[0] if angleUnit != [] else self._units['angle']

      objectUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(u.stars)]
      self._units['object'] = objectUnit[0] if objectUnit != [] else u.stars

      densityUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(u.stars/u.m**3)]
      densityUnit2 = [this for this in UNIT_SYSTEM if this.is_equivalent(1/u.m**3)]
      if densityUnit == [] and densityUnit2 != []: 
        densityUnit = [self._units['object'] * densityUnit2[0]]
      elif densityUnit == [] and objectUnit != [] and lengthUnit != []: 
        densityUnit = [self._units['object']/self._units['length']**3]
      elif densityUnit == [] and densityUnit2 == [] and objectUnit != []:
         densityLength = [this for this in self._units['density'].bases if this.is_equivalent(u.m)][0]
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

    star_b = verify_unit(star_b, u.au)
    star_v = verify_unit(star_v, u.km/u.s)

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

    star_q = verify_unit(star_q, u.au)
    star_vinf = verify_unit(star_v, u.km/u.s)
    return 1 + star_q * star_vinf * star_vinf / mu

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
    mu = G * (sim.calculate_com().m * units['mass'] + star.m)

    star_e = vinf_and_b_to_e(mu, star.b, star.v)
    return star.b * _numpy.sqrt((star_e - 1.0)/(star_e + 1.0))

def determine_eccentricity(sim, star_mass, star_b, star_v=None, star_e=None):
    '''Calculate the eccentricity of the flyby star. '''
    # Convert the units of the REBOUND Simulation into Astropy Units.
    units = rebound_units(sim)
    G = (sim.G * units['length']**3 / units['mass'] / units['time']**2)
    star_mass = verify_unit(star_mass, units['mass'])
    star_b = verify_unit(star_b, units['length'])
    star_v = verify_unit(star_v, units['length']/units['time'])

    mu = G * (sim.calculate_com().m * units['mass'] + star_mass)
    if star_e is not None and star_v is not None: raise AssertionError('Overdetermined. Cannot specify an eccentricity and a velocity for the perturbing star.')
    elif star_e is not None and star_v is None:
        # Simply use the eccentricity if it is defined.
        return star_e
    elif star_e is None and star_v is not None:
        # If `star_v` is defined convert it to eccentricity.
        # Assumes that `star_v` is in units of km/s.
        return vinf_and_b_to_e(mu=mu, star_b=star_b, star_v=star_v)
    else: raise AssertionError('Undetermined. Specify either an eccentricity or a velocity for the perturbing star.')


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
    sun_mass = 1 * u.solMass # mass of the Sun in units of Msun

    v = verify_unit(v, unit_set.units['velocity'])
    R = verify_unit(R, unit_set.units['length'])
    M = verify_unit(M, unit_set.units['mass'])

    return (_numpy.pi * R**2) * (1 + 2*G*(sun_mass + M)/(R * v**2))

def encounter_rate(n, v, R, M=(1 * u.solMass), unit_set=UnitSet()):
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
    defrebunits = {'length': u.au, 'mass': u.solMass, 'time': u.yr2pi}
    simunits = sim.units
    
    for unit in simunits:
        if simunits[unit] == None: simunits[unit] = defrebunits[unit]
        else: simunits[unit] = u.Unit(simunits[unit])
    return simunits

def verify_unit(value, unit):
    return value.to(unit) if isQuantity(value) else value * unit

def isList(l):
    '''Determines if an object is a list or numpy array. Used for flyby parallelization.'''
    if isinstance(l,(list,_numpy.ndarray)): return True
    else: return False

def isQuantity(var):
    '''Determines if an object is an Astropy Quantity. Used for Stellar Environment initializations.'''
    return isinstance(var, u.quantity.Quantity)

def isUnit(var):
    '''Determines if an object is an Astropy Quantity. Used for Stellar Environment initializations.'''
    return isinstance(var, (u.core.IrreducibleUnit, u.core.CompositeUnit, u.Unit))


############################################################
###################### Exceptions ##########################
############################################################

class InvalidKeyException(Exception):
  def __init__(self): super().__init__('Invalid key type.')

class InvalidUnitException(Exception):
  def __init__(self): super().__init__('Value is not a valid unit type.')

