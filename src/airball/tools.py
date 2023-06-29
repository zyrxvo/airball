from lib2to3.pytree import convert
import numpy as _numpy
import rebound as _rebound

from scipy.special import erfinv as _erfinv
from scipy.stats import uniform as _uniform
import astropy.constants as const
from . import units

################################
###### Useful Constants ########
################################

twopi = 2.*_numpy.pi
# yr2pi = u.def_unit('yrtwopi', u.yr/twopi, format={'latex': r'(yr/2\pi)'})

# convert_kms_to_auyr2pi = 0.03357365989646266 # 1 km/s to AU/Yr2Pi
# convert_kms_to_auyr = 0.2109495265696987 # 1 km/s to AU/Yr
# convert_auyr2pi_to_kms = 1.0/convert_kms_to_auyr2pi

# convert_pc3_to_au3 = ((_numpy.pi**3)/272097792000000000) # 1 parsec^-3 to AU^-3
# convert_au3_to_pc3 = 1.0/convert_pc3_to_au3

############################################################
### Stellar Mass generators using Initial Mass Functions ###
############################################################

# def imf_gen_1(size):
#     '''
#         Generate stellar mass samples for single star systems between 0.01 and 1.0 Solar Mass.
        
#         Computed using the inverted cumulative probability distribution (CDF) from the initial mass function (IMF) given in equation (17) by Chabrier (2003) https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C/abstract

#         Parameters
#         ----------
#         size: the number of samples to draw.
#     '''
#     n = int(size)
#     u = _uniform.rvs(size=n)
#     return 0.08*_numpy.exp(2.246879476250902 * _erfinv(-0.8094067254228074 + 1.6975098420629455*u))

# def imf_gen_10(size):
#     '''
#         Generate stellar mass samples for single star systems between 0.01 and 10.0 Solar Masses.
        
#         Computed using the inverted cumulative probability distribution (CDF) by smoothly combining the initial mass function (IMF) given in equation (17) by Chabrier (2003) https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C/abstract for stars less than 1.0 Solar Mass with the standard power-law IMF from Salpeter (1955) https://ui.adsabs.harvard.edu/abs/1955ApJ...121..161S/abstract for stars more than 1.0 Solar Mass.

#         Parameters
#         ----------
#         size: the number of samples to draw.
#     '''
#     n = int(size)
#     u = _uniform.rvs(size=n)
#     return _numpy.where(u > 0.9424222533172513, 0.11575164791201686 / (1.0030379829867349 - u)**(10/13.), 0.08*_numpy.exp(2.246879476250902 * _erfinv(-0.8094067254228074 + 1.801220032833315*u)))

# def imf_gen_100(size):
#     '''
#         Generate stellar mass samples for single star systems between 0.01 and 100.0 Solar Masses.
        
#         Computed using the inverted cumulative probability distribution (CDF) by smoothly combining the initial mass function (IMF) given in equation (17) by Chabrier (2003) https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C/abstract for stars less than 1.0 Solar Mass with the standard power-law IMF from Salpeter (1955) https://ui.adsabs.harvard.edu/abs/1955ApJ...121..161S/abstract for stars more than 1.0 Solar Mass.

#         Parameters
#         ----------
#         size: the number of samples to draw.
#     '''
#     n = int(size)
#     u = _uniform.rvs(size=n)
#     return _numpy.where(u > 0.9397105089399359, 0.11549535807627886 / (1.0001518217134586 - u)**(10/13.), 0.08*_numpy.exp(2.246879476250902 * _erfinv(-0.8094067254228074 + 1.8064178551944312*u)))




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

def cross_section(star_mass, R, v):
    '''
        The cross-section with gravitational focusing.
        
        Parameters
        ----------
        star_mass : the mass of flyby star in units of Msun
        R : the maximum interaction radius in units of AU
        v : the typical velocity from the distribution in units of AU/Yr
    '''
    G = const.G.decompose([units.AU, units.yr, units.solMass]) # Newton's gravitational constant in units of Msun, AU, and Years (G ~ 4π^2).
    sun_mass = 1 * units.solMass # mass of the Sun in units of Msun
    return (_numpy.pi * R**2) * (1 + 2*G*(sun_mass + star_mass)/(R * v**2))

def encounter_rate(n, vbar, R, star_mass=(1 * units.solMass)):
    '''
        The expected flyby encounter rate within an stellar environment
        
        Parameters
        ----------
        n : stellar number density in units of AU^{-3}
        vbar : velocity dispersion in units of km/s
        R : interaction radius in units of AU
        star_mass : mass of a typical flyby star in units of Msun
    '''
    # vv = vbar*convert_kms_to_auyr # Convert from km/s to AU/yr
    # Include factor of sqrt(2) in cross-section to account for relative velocities at infinity.
    return n * vbar * cross_section(star_mass, R, _numpy.sqrt(2.)*vbar)


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
