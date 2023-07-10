from lib2to3.pytree import convert
import numpy as _numpy
import rebound as _rebound
import astropy.constants as const
from . import units

twopi = 2.*_numpy.pi

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
      if densityUnit == [] and densityUnit2 != []: 
        densityUnit = [self._units['object'] * densityUnit2[0]]
      elif densityUnit == [] and objectUnit != [] and lengthUnit != []: 
        densityUnit = [self._units['object']/self._units['length']**3]
      elif densityUnit == [] and densityUnit2 == [] and objectUnit != []:
         densityLength = [this for this in self._units['density'].bases if this.is_equivalent(units.m)][0]
         densityUnit = [self._units['object']/densityLength**3]
      self._units['density'] = densityUnit[0] if densityUnit != [] else self._units['density']
    
    self._UNIT_SYSTEM = list(self._units.values())



############################################################
################### Helper Functions #######################
############################################################

def rebound_units(sim):
    defrebunits = {'length': units.au, 'mass': units.solMass, 'time': units.yr2pi}
    siunits = {'length': units.m, 'mass': units.kg, 'time': units.s}
    rebunitset = {'length': _rebound.units.lengths_SI, 'mass': _rebound.units.masses_SI, 'time': _rebound.units.times_SI}
    simunits = sim.units
    
    for unit in simunits:
        if simunits[unit] == None: simunits[unit] = defrebunits[unit]
        else: simunits[unit] = rebunitset[unit][simunits[unit]] * siunits[unit]
    return simunits

# Implemented from StackOverflow: https://stackoverflow.com/a/14314054
def moving_average(a, n=3) :
    '''Compute the moving average of an array of numbers using the nearest n elements.'''
    ret = _numpy.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def vinf_and_b_to_e(mu, star_b, star_vinf):
    '''
        Using the impact parameter to convert from the relative velocity at infinity between the two stars to the eccentricity of the flyby star.
        Equation (2) from Spurzem et al. (2009) https://ui.adsabs.harvard.edu/abs/2009ApJ...697..458S/abstract

        Parameters
        ----------
        mu : the total mass of the system (Sun, planets, and flyby star) times the gravitational constant G
        b :  impact parameter of the flyby star
        vinf : the relative velocity at infinity between the central star and the flyby star (hyperbolic excess velocity)
    '''

    star_b = star_b.to(units.au) if isQuantity(star_b) else star_b * units.au
    star_vinf = star_vinf.to(units.km/units.s) if isQuantity(star_vinf) else star_vinf * units.km/units.s

    numerator = star_b * star_vinf**2.
    return _numpy.sqrt(1 + (numerator/mu)**2.)

def determine_eccentricity(sim, star_mass, star_b, star_v=None, star_e=None):
    '''Calculate the eccentricity of the flyby star. '''
    # Convert the units of the REBOUND Simulation into Astropy Units.
    units = rebound_units(sim)
    G = (sim.G * units['length']**3 / units['mass'] / units['time']**2)
    star_mass = star_mass.to(units['mass']) if isQuantity(star_mass) else star_mass * units['mass']
    star_b = star_b.to(units['length']) if isQuantity(star_b) else star_b * units['length']
    star_v = star_v.to(units['length']/units['time']) if isQuantity(star_v) else star_v * units['length']/units['time']

    mu = G * (_numpy.sum([p.m for p in sim.particles]) * units['mass'] + star_mass)
    if star_e is not None and star_v is not None: raise AssertionError('Overdetermined. Cannot specify an eccentricity and a velocity for the perturbing star.')
    elif star_e is not None and star_v is None:
        # Simply use the eccentricity if it is defined.
        return star_e
    elif star_e is None and star_v is not None:
        # If `star_v` is defined convert it to eccentricity.
        # Assumes that `star_v` is in units of km/s.
        return vinf_and_b_to_e(mu=mu, star_b=star_b, star_vinf=star_v)
    else: raise AssertionError('Undetermined. Specify either an eccentricity or a velocity for the perturbing star.')

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

def cross_section(M, R, v, unit_set=UnitSystem()):
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
    sun_mass = 1 * units.solMass # mass of the Sun in units of Msun

    v = verify_unit(v, unit_set.units['velocity'])
    R = verify_unit(R, unit_set.units['length'])
    M = verify_unit(M, unit_set.units['mass'])

    return (_numpy.pi * R**2) * (1 + 2*G*(sun_mass + M)/(R * v**2))

def encounter_rate(n, v, R, M=(1 * units.solMass), unit_set=UnitSystem()):
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

def verify_unit(value, unit):
    return value.to(unit) if isQuantity(value) else value * unit

def isList(l):
    '''Determines if an object is a list or numpy array. Used for flyby parallelization.'''
    if isinstance(l,(list,_numpy.ndarray)): return True
    else: return False


def isQuantity(var):
    '''Determines if an object is an Astropy Quantity. Used for Stellar Environment initializations.'''
    return isinstance(var, units.quantity.Quantity)

def isUnit(var):
    '''Determines if an object is an Astropy Quantity. Used for Stellar Environment initializations.'''
    return isinstance(var, (units.core.IrreducibleUnit, units.core.CompositeUnit))
