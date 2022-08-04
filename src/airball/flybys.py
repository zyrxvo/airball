import rebound as _rebound
import numpy as _numpy

from scipy.special import j0 as _j0,jv as _jv
from scipy.special import erfinv as _erfinv
from scipy.stats import uniform as _uniform

################################
###### Useful Constants ########
################################

_twopi = 2.*_numpy.pi
_kms_to_auyr2pi = 0.03357365989646266 # 1 km/s to AU/Yr2Pi
_kms_to_auyr = 0.2109495265696987 # 1 km/s to AU/Yr


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
    v = star_vinf*_kms_to_auyr2pi # Convert velocity units from km/s to AU/Yr2Pi
    # mu = G
    numerator = star_b * v**2.
    return _numpy.sqrt(1 + (numerator/mu)**2.)


def cross_section(star_mass, R, v):
    '''
        The cross-section with gravitational focusing.
        
        Parameters
        ----------
        star_mass : the mass of flyby star in units of Msun
        R : the maximum interaction radius in units of AU
        v : the typical velocity from the distribution in units of AU/Yr
    '''
    
    G = _twopi**2 # Newton's gravitational constant in units of Msun, AU, and Years
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
    vv = vbar*_kms_to_auyr # Convert from km/s to AU/yr
    # Include factor of sqrt(2) in cross-section to account for relative velocities.
    return n * vv * cross_section(star_mass, R, _numpy.sqrt(2.)*vv)


############################################################
#################### Flyby Functions #######################
############################################################

def flyby_star(sim, star_mass=1, star_b=100, star_v=None,  star_e=None, star_omega='uniform', star_Omega='uniform', star_inc='uniform', star_rmax=1e6):
    '''
        Return a REBOUND Particle for a flyby star given a REBOUND Simulation and flyby parameters.
        
        Parameters
        ----------
        sim : the REBOUND Simulation (star and planets) that will experience the flyby star
        star_mass : the mass of the flyby star in units of Msun
        star_b : impact parameter of the flyby star in units of AU
        star_v : the relative velocity at infinity between the central star and the flyby star (hyperbolic excess velocity) in units of km/s. Only specify star_v OR star_e, not both.
        star_e : the eccentricity of the flyby star (e > 1). Only specify star_e OR star_v, not both.
        star_omega : the argument of periapsis of the flyby star
        star_Omega : the longitude of the ascending node of the flyby star
        star_inc : the inclination of the flyby star
        star_rmax : the starting distance of the flyby star in units of AU
    '''
    
    mu = sim.G * (_numpy.sum([p.m for p in sim.particles]) + star_mass)
    if star_e is None and star_v is not None:
        # If `star_v` is defined convert it to eccentricity.
        # Assumes that `star_v` is in units of km/s.
        e = vinf_and_b_to_e(mu=mu, star_b=star_b, star_vinf=star_v)
    elif star_e is not None and star_v is None:
        # Simply use the eccentricity if it is defined.
        e = star_e
    elif star_e is not None and star_v is not None: raise AssertionError('Cannot specify an eccentricity and a velocity for the perturbing star.')
    else: raise AssertionError('Specify either an eccentricity or a velocity for the perturbing star.')
    
    #################################################
    ## Calculation of Flyby Star Initial Conditions ## 
    #################################################
    
    # Calculate the orbital elements of the flyby star.
    rmax = star_rmax # This is the starting distance of the flyby star in AU
    a = -star_b/_numpy.sqrt(e**2. - 1.) # Compute the semi-major axis of the flyby star
    l = -a*(e*e-1.) # Compute the semi-latus rectum of the hyperbolic orbit (-a because the semi-major axis is negative)
    f = _numpy.arccos((l/rmax-1.)/e) # Compute the true anomaly

    #################################################

    return _rebound.Particle(sim, m=star_mass, a=a, e=e, f=-f, omega=star_omega, Omega=star_Omega, inc=star_inc, hash='flybystar')


def flyby(sim, star_mass=1, star_b=100, star_v=None,  star_e=None, star_omega='uniform', star_Omega='uniform', star_inc='uniform', star_rmax=1e6, showFlybySetup=False, returnOneOrbitSlice=False, axOrbitSlice=None, removeFlybyStar=True, closeEncounterEstimate=False):
    '''
        Simulate a stellar flyby to a REBOUND simulation.
        
        Because REBOUND Simulations are C structs underneath the Python, this function passes the simulation by reference. 
        Any changes made inside this function to the REBOUND simulation are permanent.
        This function assumes that you are using a WHFAST integrator with REBOUND.
        
        Parameters
        ----------
        sim : the REBOUND Simulation (star and planets) that will experience the flyby star
        star_mass : the mass of the flyby star in units of Msun
        star_b : impact parameter of the flyby star in units of AU
        star_v : the relative velocity at infinity between the central star and the flyby star (hyperbolic excess velocity) in units of km/s. Only specify star_v OR star_e, not both.
        star_e : the eccentricity of the flyby star (e > 1). Only specify star_e OR star_v, not both.
        star_omega : the argument of periapsis of the flyby star
        star_Omega : the longitude of the ascending node of the flyby star
        star_inc : the inclination of the flyby star
        star_rmax : the starting distance of the flyby star in units of AU
        showFlybySetup : True or False. Shows a REBOUND OrbitPlot snapshot of the system when the flyby star is at periapsis.
    '''
    
    mu = sim.G * (_numpy.sum([p.m for p in sim.particles]) + star_mass)
    if star_e is None and star_v is not None:
        # If `star_v` is defined convert it to eccentricity.
        # Assumes that `star_v` is in units of km/s.
        sun_mass = sim.particles[0].m
        planet_mass = sim.particles[1].m
        e = vinf_and_b_to_e(mu=mu, star_b=star_b, star_vinf=star_v)
    elif star_e is not None and star_v is None:
        # Simply use the eccentricity if it is defined.
        e = star_e
    elif star_e is not None and star_v is not None: raise AssertionError('Cannot specify an eccentricity and a velocity for the perturbing star.')
    else: raise AssertionError('Specify either an eccentricity or a velocity for the perturbing star.')
    
    #################################################
    ## Calculation of Flyby Star Initial Conditions ## 
    #################################################
    
    # Calculate the orbital elements of the flyby star.
    rmax = star_rmax # This is the starting distance of the flyby star in AU
    a = -star_b/_numpy.sqrt(e**2. - 1.) # Compute the semi-major axis of the flyby star
    l = -a*(e*e-1.) # Compute the semi-latus rectum of the hyperbolic orbit (-a because the semi-major axis is negative)
    f = _numpy.arccos((l/rmax-1.)/e) # Compute the true anomaly
    mu = sim.G * _numpy.sum([p_j.m for p_j in sim.particles])
    
    # # Calculate half of the integration time for the flyby star.
    # E = _numpy.arccosh((_numpy.cos(f)+e)/(1.+e*_numpy.cos(f))) # Compute the eccentric anomaly
    # M = e * _numpy.sinh(E)-E # Compute the mean anomaly
    # tperi = M/_numpy.sqrt(mu/(-a*a*a)) # Compute the time to periapsis (-a because the semi-major axis is negative)

    #################################################
    
    # Add the flyby star to the simulation. 
    sim.move_to_hel() # Move the system into the heliocentric frame of reference.
    sim.add(m=star_mass, a=a, e=e, f=-f, omega=star_omega, Omega=star_Omega, inc=star_inc, hash='flybystar')
    sim.ri_whfast.recalculate_coordinates_this_timestep = 1 # Because a new particle was added, we need to tell REBOUND to recalculate the coordinates.
    sim.move_to_com() # Move the system back into the centre of mass/momentum frame for integrating.

    tperi = sim.particles['flybystar'].T # Compute the time to periapsis.
    
    de = None
    # Integrate the flyby. Start at the current time and go to twice the time to periapsis.
    if showFlybySetup:
        t1 = sim.t + tperi
        t2 = sim.t + 2*tperi
        sim.integrate(t1)
        print('During the Flyby')
        sim.move_to_hel()
        _rebound.OrbitPlot(sim, color=True, slices=True, periastron=True);
        if returnOneOrbitSlice and axOrbitSlice is not None:
            _rebound.plotting.OrbitPlotOneSlice(sim, axOrbitSlice, color=True, orbit_type='trail')
        sim.move_to_com()
        sim.integrate(t2)
    elif returnOneOrbitSlice and axOrbitSlice is not None:
        t1 = sim.t + tperi
        t2 = sim.t + 2*tperi
        sim.integrate(t1)
        sim.move_to_hel()
        _rebound.plotting.OrbitPlotOneSlice(sim, axOrbitSlice, color=True, orbit_type='trail')
        sim.move_to_com()
        sim.integrate(t2)
    elif closeEncounterEstimate:
        t1 = sim.t + tperi
        t2 = sim.t + 2*tperi
        sim.integrate(t1)
        de = energy_change_close_encounters_sim(sim)
        sim.move_to_com()
        sim.integrate(t2)
    else:
        sim.integrate(sim.t + 2.*tperi)
    
    if removeFlybyStar:
        # Remove the flyby star. 
        sim.remove(hash='flybystar')
        sim.ri_whfast.recalculate_coordinates_this_timestep = 1 # Because a particle was removed, we need to tell REBOUND to recalculate the coordinates.
        sim.move_to_com() # Readjust the system back into the centre of mass/momentum frame for integrating.
    
    if closeEncounterEstimate and de is not None:
        return de



############################################################
################## Analytical Estimates ####################
############################################################

def binary_energy(sun_mass, planet_mass, planet_a):
    '''
        The energy of a binary system, -(G*M*m)/(2*a).
        
        Parameters
        ----------
        sun_mass : the mass of the central star in units of Msun
        planet_mass : the mass of the planet in units of Msun
        planet_a :  the semi-major axis of the planet in units of AU
    '''
    G = 1.
    return -G * sun_mass * planet_mass / (2.*planet_a)
    
def energy_change_adiabatic_estimate(sun_mass=1, planet_mass=5e-5, planet_a=30, planet_e=1e-3, star_mass=1, star_b=100, star_v=None,  star_e=None, star_omega='uniform', star_Omega='uniform', star_inc='uniform', t0=0, averaged=True):
    '''
        An analytical estimate for the change in energy of a binary system due to a flyby star.
        
        From the conclusions of Roy & Haddow (2003) https://ui.adsabs.harvard.edu/abs/2003CeMDA..87..411R/abstract and Heggie (2006) https://ui.adsabs.harvard.edu/abs/2006fbp..book...20H/abstract. The orbital element angles of the flyby star are determined with respect to the plane defined by the binary orbit. In REBOUND this is the same as when the inclination of the planet is zero.
        
        Parameters
        ----------
        sun_mass : the mass of the central star in units of Msun
        planet_mass : the mass of the planet (binary object) in units of Msun
        planet_a : the semi-major axis of the planet in units of AU
        planet_e : the eccentricity of the planet
        star_mass : the mass of the flyby star in units of Msun
        star_b :  impact parameter of the flyby star in units of AU
        star_v : the relative velocity at infinity between the central star and the flyby star (hyperbolic excess velocity) in units of km/s. Only specify star_v OR star_e, not both.
        star_e : the eccentricity of the flyby star (e > 1). Only specify star_e OR star_v, not both.
        star_omega : the argument of periapsis of the flyby star
        star_Omega : the longitude of the ascending node of the flyby star
        star_inc : the inclination of the flyby star
        t0 :  the time of periapsis passage of the planet where t=0 is when the flyby star passes perihelion
    '''

    G = 1 # Newton's Gravitational constant in units of Msun, AU, Yr2Pi
    m1, m2, m3 = sun_mass, planet_mass, star_mass # redefine the masses for convenience
    M12 = m1 + m2 # total mass of the binary system
    M123 = m1 + m2 + m3 # total mass of all the objects involved
    
    if star_e is None and star_v is not None:
        # If `star_v` is defined convert it to eccentricity.
        # Assumes that `star_v` is in units of km/s.
        mu = G*M123
        es = vinf_and_b_to_e(mu=mu, star_b=star_b, star_vinf=star_v)
    elif star_e is not None and star_v is None:
        # Simply use the eccentricity if it is defined.
        es = star_e
    elif star_e is not None and star_v is not None: raise AssertionError('Cannot specify an eccentricity and a velocity for the perturbing star.')
    else: raise AssertionError('Specify either an eccentricity or a velocity for the perturbing star.')
    
    a, e = planet_a, planet_e # redefine the orbital elements of the planet for convenience
    b = a*_numpy.sqrt(1-e**2) # compute the semi-minor axis of the planet
    n = _numpy.sqrt(G*M12/a**3) # compute the mean motion of the planet
    
    # If the orientation of the flyby star is random, then sample from uniform distributions.
    if star_omega == 'uniform': omega = _numpy.random.uniform(-_numpy.pi, _numpy.pi)
    else: omega = star_omega
    if star_Omega == 'uniform': Omega = _numpy.random.uniform(-_numpy.pi, _numpy.pi)
    else: Omega = star_Omega
    if star_inc == 'uniform': inc = _numpy.random.uniform(-_numpy.pi, _numpy.pi)
    else: inc = star_inc

    w, W, i = omega, Omega, inc # redefine the orientation elements of the flyby star for convenience
    V = star_v * _kms_to_auyr2pi # convert the velocity of the star to standard REBOUND units
    GM123 = G*M123 
    q = (- GM123 + _numpy.sqrt( GM123**2. + star_b**2. * V**4.))/V**2. # compute the periapsis of the flyby star

    # Calculate the following convenient functions of the planet's eccentricity and Bessel functions of the first kind of order n.
    e1 = _jv(-1,e) - 2*e*_j0(e) + 2*e*_jv(2,0) - _jv(3,e)
    e2 = _jv(-1,e) - _jv(3,e)
    e4 = _jv(-1,e) - e*_j0(e) - e*_jv(2,e) + _jv(3,e)

    # Calculate a convenient function of the planet's semi-major axis and the flyby star's periapsis.
    k = _numpy.sqrt((2*M12*q**3)/(M123*a**3))
    
    # Calculate convenient functions of the flyby star's eccentricity.
    f1 = ((es + 1.0)**(0.75)) / ((2.0**(0.75)) * (es*es))
    f2 = (3.0/(2.0*_numpy.sqrt(2.0))) * (_numpy.sqrt((es*es) - 1.0) - _numpy.arccos(1.0/es)) / ((es - 1.0)**(1.5))
    
    # Compute the prefactor and terms of the calculation done by Roy & Haddow (2003)
    prefactor = (-_numpy.sqrt(_numpy.pi)/8.0) * ((G*m1*m2*m3)/M12) * ((a*a)/(q*q*q)) * f1 * k**(2.5) * _numpy.exp((-2.0*k/3.0)*f2)
    term1 = e1 * ( _numpy.sin(2.0*w + n*t0)*_numpy.cos(2.0*i - 1.0)- _numpy.sin(2.0*w + n*t0)*_numpy.cos(2.0*i)*_numpy.cos(2.0*W) - 3.0*_numpy.sin(n*t0 + 2.0*w)*_numpy.cos(2.0*W) - 4.0*_numpy.sin(2.0*W)*_numpy.cos(2.0*w + n*t0)*_numpy.cos(i) )
    term2 = e2 * (1.0 - e*e) * ( _numpy.sin(2.0*w + n*t0)*(1.0-_numpy.cos(2.0*i)) - _numpy.sin(2.0*w + n*t0)*_numpy.cos(2.0*i)*_numpy.cos(2.0*W) - 3.0*_numpy.sin(n*t0 +2.0*w)*_numpy.cos(2.0*W) - 4.0*_numpy.cos(n*t0 + 2.0*w)*_numpy.sin(2.0*W)*_numpy.cos(i) )
    term3 = e4 * _numpy.sqrt(1.0 - e*e) * (-2.0*_numpy.cos(2.0*i)*_numpy.cos(2.0*w + n*t0)*_numpy.sin(2.0*W) - 6.0*_numpy.cos(2.0*w + n*t0)*_numpy.sin(2.0*W) - 8.0*_numpy.cos(2.0*W)*_numpy.sin(2.0*w + n*t0)*_numpy.cos(i) )
    
    if averaged: return prefactor
    else: return prefactor * ( term1 + term2 + term3)

def relative_energy_change(sun_mass, planet_mass, planet_a, planet_e, star_mass, star_b, star_v, star_e=None, star_omega='uniform', star_Omega='uniform', star_inc='uniform', t0=0, averaged=False):
    '''
        An analytical estimate for the relative change in energy of a binary system due to a flyby star.
        
        Combines energy_change_adiabatic_estimate(...) and binary_energy(...) functions.
        
        Parameters
        ----------
        sun_mass : the mass of the central star in units of Msun
        planet_mass : the mass of the planet (binary object) in units of Msun
        planet_a : the semi-major axis of the planet in units of AU
        planet_e : the eccentricity of the planet
        star_mass : the mass of the flyby star in units of Msun
        star_b :  impact parameter of the flyby star in units of AU
        star_v : the relative velocity at infinity between the central star and the flyby star (hyperbolic excess velocity) in units of km/s.
    '''
    return energy_change_adiabatic_estimate(sun_mass=sun_mass, planet_mass=planet_mass, planet_a=planet_a, planet_e=planet_e, star_mass=star_mass, star_b=star_b, star_v=star_v, star_e=star_e, star_omega=star_omega, star_Omega=star_Omega, star_inc=star_inc, t0=t0, averaged=averaged)/binary_energy(sun_mass, planet_mass, planet_a)

def eccentricity_change_adiabatic_estimate(sun_mass=1, planet_mass=5e-05, planet_a=30, planet_e=0.001, star_mass=1, star_b=100, star_v=None, star_e=None, star_omega='uniform', star_Omega='uniform', star_inc='uniform', averaged=False):
    '''
        An analytical estimate for the change in eccentricity of an eccentric binary system due to a flyby star.
        
        From the conclusions of Heggie & Rasio (1996) Equation (A3) from Spurzem et al. (2009) https://ui.adsabs.harvard.edu/abs/2009ApJ...697..458S/abstract. 
        The orbital element angles of the flyby star are determined with respect to the plane defined by the binary orbit. In REBOUND this is the same as when the inclination of the planet is zero.
        
        Parameters
        ----------
        sun_mass : the mass of the central star in units of Msun
        planet_mass : the mass of the planet (binary object) in units of Msun
        planet_a : the semi-major axis of the planet in units of AU
        planet_e : the eccentricity of the planet
        star_mass : the mass of the flyby star in units of Msun
        star_b :  impact parameter of the flyby star in units of AU
        star_v : the relative velocity at infinity between the central star and the flyby star (hyperbolic excess velocity) in units of km/s. Only specify star_v OR star_e, not both.
        star_e : the eccentricity of the flyby star (e > 1). Only specify star_e OR star_v, not both.
        star_omega : the argument of periapsis of the flyby star
        star_Omega : the longitude of the ascending node of the flyby star
        star_inc : the inclination of the flyby star
    '''
    G = 1 # Newton's Gravitational constant in units of Msun, AU, Yr2Pi
    m1, m2, m3 = sun_mass, planet_mass, star_mass # redefine the masses for convenience
    M12 = m1 + m2 # total mass of the binary system
    M123 = m1 + m2 + m3 # total mass of all the objects involved
    
    if star_e is None and star_v is not None:
        # If `star_v` is defined convert it to eccentricity.
        # Assumes that `star_v` is in units of km/s.
        mu = G*M123
        es = vinf_and_b_to_e(mu=mu, star_b=star_b, star_vinf=star_v)
    elif star_e is not None and star_v is None:
        # Simply use the eccentricity if it is defined.
        es = star_e
    elif star_e is not None and star_v is not None: raise AssertionError('Cannot specify an eccentricity and a velocity for the perturbing star.')
    else: raise AssertionError('Specify either an eccentricity or a velocity for the perturbing star.')
    
    a, e = planet_a, planet_e # redefine the orbital elements of the planet for convenience
    b = a*_numpy.sqrt(1-e**2) # compute the semi-minor axis of the planet
    n = _numpy.sqrt(G*M12/a**3) # compute the mean motion of the planet
    
    # If the orientation of the flyby star is random, then sample from uniform distributions.
    if star_omega == 'uniform': omega = _numpy.random.uniform(-_numpy.pi, _numpy.pi)
    else: omega = star_omega
    if star_Omega == 'uniform': Omega = _numpy.random.uniform(-_numpy.pi, _numpy.pi)
    else: Omega = star_Omega
    if star_inc == 'uniform': inc = _numpy.random.uniform(-_numpy.pi, _numpy.pi)
    else: inc = star_inc

    w, W, i = omega, Omega, inc # redefine the orientation elements of the flyby star for convenience
    V = star_v * _kms_to_auyr2pi # convert the velocity of the star to standard REBOUND units
    GM123 = G*M123 
    q = (- GM123 + _numpy.sqrt( GM123**2. + star_b**2. * V**4.))/V**2. # compute the periapsis of the flyby star
    
    prefactor = (-15.0/4.0) * m3 / _numpy.sqrt(M12*M123) * ((a/q)**1.5) * ((e * _numpy.sqrt(1.0 - e*e))/((1.0 + es)**1.5))
    t1 = _numpy.sin(i) * _numpy.sin(i) * _numpy.sin(2.0*W) * ( _numpy.arccos(-1.0/es) + _numpy.sqrt(es*es - 1.0) )
    t2 = (1.0/3.0) * (1.0 + _numpy.cos(i)*_numpy.cos(i)) * _numpy.cos(2.0*w) * _numpy.sin(2.0*W)
    t3 = 2.0 * _numpy.cos(i) * _numpy.sin(2.0*w) * _numpy.cos(2.0*W) * ((es*es - 1.0)**1.5)/(es*es)
    
    if averaged: return prefactor * ( ( _numpy.arccos(-1.0/es) + _numpy.sqrt(es*es - 1.0) ) + (2.0/3.0) + (2.0 * ((es*es - 1.0)**1.5)/(es*es)) )
    else: return prefactor * (t1 + t2 + t3)

def energy_change_close_encounters_sim(sim):
    '''
        An analytical estimate for the change in energy of a binary system due to a flyby star.
        
        From the conclusions of Roy & Haddow (2003) https://ui.adsabs.harvard.edu/abs/2003CeMDA..87..411R/abstract and Heggie (2006) https://ui.adsabs.harvard.edu/abs/2006fbp..book...20H/abstract. The orbital element angles of the flyby star are determined with respect to the plane defined by the binary orbit. In REBOUND this is the same as when the inclination of the planet is zero.
        
        Parameters
        ----------
        sim : three-body REBOUND sim
    '''
    sim.move_to_hel()
    p = sim.particles

    G = sim.G # Newton's Gravitational constant in units of Msun, AU, Yr2Pi
    m1, m2, m3 = p[0].m, p[1].m, p[2].m # redefine the masses for convenience
    M12 = m1 + m2 # total mass of the binary system
    M23 = m2 + m3 # total mass of the binary system
    M123 = m1 + m2 + m3 # total mass of all the objects involved

    V = p[2].v # velocity of the star
    es = p[2].e
    GM123 = G*M123
    b = -p[2].a * _numpy.sqrt(es*es - 1.0)
    q = (- GM123 + _numpy.sqrt( GM123**2. + b**2. * V**4.))/V**2. # compute the periapsis of the flyby star

    x,y,z = p[2].xyz
    vx,vy,vz = p[1].vxyz

    cosϕ = 1.0/_numpy.sqrt(1.0 + ((q**2.0)*(V**4.0))/(M23**2.0))
    
    prefactor = (-2.0 * m1 * m2 * m3)/(M12 * M23) * V * cosϕ
    t1 = -(x*vx + y*vy + z*vz)
    t2 = (m3 * V * cosϕ)/M23
    
    sim.move_to_hel()
    
    return prefactor * (t1 + t2)
