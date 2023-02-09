from lib2to3.pytree import convert
import numpy as _numpy

from scipy.special import erfinv as _erfinv
from scipy.stats import uniform as _uniform
# import astropy.constants as const
# import astropy.units as u

################################
###### Useful Constants ########
################################

twopi = 2.*_numpy.pi

convert_kms_to_auyr2pi = 0.03357365989646266 # 1 km/s to AU/Yr2Pi
convert_kms_to_auyr = 0.2109495265696987 # 1 km/s to AU/Yr
convert_auyr2pi_to_kms = 1.0/convert_kms_to_auyr2pi

convert_pc3_to_au3 = ((_numpy.pi**3)/272097792000000000) # 1 parsec^-3 to AU^-3
convert_au3_to_pc3 = 1.0/convert_pc3_to_au3

############################################################
### Stellar Mass generators using Initial Mass Functions ###
############################################################

def imf_gen_1(size):
    '''
        Generate stellar mass samples for single star systems between 0.01 and 1.0 Solar Mass.
        
        Computed using the inverted cumulative probability distribution (CDF) from the initial mass function (IMF) given in equation (17) by Chabrier (2003) https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C/abstract

        Parameters
        ----------
        size: the number of samples to draw.
    '''
    n = int(size)
    u = _uniform.rvs(size=n)
    return 0.08*_numpy.exp(2.246879476250902 * _erfinv(-0.8094067254228074 + 1.6975098420629455*u))

def imf_gen_10(size):
    '''
        Generate stellar mass samples for single star systems between 0.01 and 10.0 Solar Masses.
        
        Computed using the inverted cumulative probability distribution (CDF) by smoothly combining the initial mass function (IMF) given in equation (17) by Chabrier (2003) https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C/abstract for stars less than 1.0 Solar Mass with the standard power-law IMF from Salpeter (1955) https://ui.adsabs.harvard.edu/abs/1955ApJ...121..161S/abstract for stars more than 1.0 Solar Mass.

        Parameters
        ----------
        size: the number of samples to draw.
    '''
    n = int(size)
    u = _uniform.rvs(size=n)
    return _numpy.where(u > 0.9424222533172513, 0.11575164791201686 / (1.0030379829867349 - u)**(10/13.), 0.08*_numpy.exp(2.246879476250902 * _erfinv(-0.8094067254228074 + 1.801220032833315*u)))

def imf_gen_100(size):
    '''
        Generate stellar mass samples for single star systems between 0.01 and 100.0 Solar Masses.
        
        Computed using the inverted cumulative probability distribution (CDF) by smoothly combining the initial mass function (IMF) given in equation (17) by Chabrier (2003) https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C/abstract for stars less than 1.0 Solar Mass with the standard power-law IMF from Salpeter (1955) https://ui.adsabs.harvard.edu/abs/1955ApJ...121..161S/abstract for stars more than 1.0 Solar Mass.

        Parameters
        ----------
        size: the number of samples to draw.
    '''
    n = int(size)
    u = _uniform.rvs(size=n)
    return _numpy.where(u > 0.9397105089399359, 0.11549535807627886 / (1.0001518217134586 - u)**(10/13.), 0.08*_numpy.exp(2.246879476250902 * _erfinv(-0.8094067254228074 + 1.8064178551944312*u)))




############################################################
################### Helper Functions #######################
############################################################

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
        mu : the total mass of the system (Sun, planets, and flyby star) times the gravitational constant G in units where G = 1, Msun, AU, Yr2Pi
        b :  impact parameter of the flyby star in units of AU
        vinf : the relative velocity at infinity between the central star and the flyby star (hyperbolic excess velocity) in units of km/s
    '''
    G = 1 # Newton's gravitational constant in units of Msun, AU, and Yr2Pi
    v = star_vinf*convert_kms_to_auyr2pi # Convert velocity units from km/s to AU/Yr2Pi
    # mu = G
    numerator = star_b * v**2.
    return _numpy.sqrt(1 + (numerator/mu)**2.)

def determine_eccentricity(sim, star_mass, star_b, star_v=None, star_e=None):
    # Calculate the eccentricity of the flyby star.
    mu = sim.G * (_numpy.sum([p.m for p in sim.particles]) + star_mass)
    if star_e is not None and star_v is not None: raise AssertionError('Overdetermined. Cannot specify an eccentricity and a velocity for the perturbing star.')
    elif star_e is not None and star_v is None:
        # Simply use the eccentricity if it is defined.
        return star_e
    elif star_e is None and star_v is not None:
        # If `star_v` is defined convert it to eccentricity.
        # Assumes that `star_v` is in units of km/s.
        sun_mass = sim.particles[0].m
        planet_mass = sim.particles[1].m
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
    
    G = twopi**2 # Newton's gravitational constant in units of Msun, AU, and Years
    sun_mass = 1 # mass of the Sun in units of Msun
    return (_numpy.pi * R**2) * (1 + 2*G*(sun_mass + star_mass)/(R * v**2))

def encounter_rate(n, vbar, R, star_mass=1):
    '''
        The expected flyby encounter rate within an stellar environment
        
        Parameters
        ----------
        n : stellar number density in units of AU^{-3}
        vbar : velocity dispersion in units of km/s
        R : interaction radius in units of AU
        star_mass : mass of a typical flyby star in units of Msun
    '''
    vv = vbar*convert_kms_to_auyr # Convert from km/s to AU/yr
    # Include factor of sqrt(2) in cross-section to account for relative velocities at infinity.
    return n * vv * cross_section(star_mass, R, _numpy.sqrt(2.)*vv)


def isList(l):
    if isinstance(l,(list,_numpy.ndarray)): return True
    else: return False




# UNIT_SYSTEM = [u.AU, u.yr, u.solMass]
# G_REBOUND = const.G.decompose(UNIT_SYSTEM).value
# def convertUnits(x):
#   return [xnew.decompose(UNIT_SYSTEM).value for xnew in x]

# numberdensity, velocity = convertUnits([numberdensity, velocity])
