import numpy as _numpy
import joblib as _joblib
import rebound as _rebound
from scipy.special import j0 as _j0,jv as _jv

from .tools import *
from .stars import *
from . import units as u

############################################################
################## Analytical Estimates ####################
############################################################

def binary_energy(sim, particle_index=1):#sun_mass, planet_mass, planet_a):
    '''
        The energy of a binary system, -(G*M*m)/(2*a).
        
        Parameters
        ----------
        sim : REBOUND Simulation
    '''
    index = int(particle_index)
    unit_set = rebound_units(sim)
    G = (sim.G * unit_set['length']**3 / unit_set['mass'] / unit_set['time']**2)
    p = sim.particles
    return (-G * p[0].m * unit_set['mass'] * p[index].m * unit_set['mass'] / (2. * p[index].a * unit_set['length'])).decompose(list(unit_set.values()))
    
def energy_change_adiabatic_estimate(sim, star, averaged=True, particle_index=1):
    '''
        An analytical estimate for the change in energy of a binary system due to a flyby star.
        
        From the conclusions of Roy & Haddow (2003) https://ui.adsabs.harvard.edu/abs/2003CeMDA..87..411R/abstract and Heggie (2006) https://ui.adsabs.harvard.edu/abs/2006fbp..book...20H/abstract. The orbital element angles of the flyby star are determined with respect to the plane defined by the binary orbit. In REBOUND this is the same as when the inclination of the planet is zero.
        
        Parameters
        ----------
        sim : REBOUND Simulation
        star : AIRBALL Star flyby object
    '''
    index = int(particle_index)
    unit_set = rebound_units(sim)
    t0 = 0*unit_set['time']
    G = (sim.G * unit_set['length']**3 / unit_set['mass'] / unit_set['time']**2)

    sim = sim.copy()
    sim.rotate(_rebound.Rotation.to_new_axes(newz=sim.angular_momentum()))
    
    p = sim.particles
    m1, m2, m3 = p[0].m * unit_set['mass'], p[index].m * unit_set['mass'], star.mass # redefine the masses for convenience
    M12 = m1 + m2 # total mass of the binary system
    M123 = m1 + m2 + m3 # total mass of all the objects involved
    
    mu = G * (system_mass(sim)  * unit_set['mass'] + m3)
    es = vinf_and_b_to_e(mu=mu, star_b=star.b, star_v=star.v)
    
    a, e = p[index].a * unit_set['length'], p[index].e # redefine the orbital elements of the planet for convenience
    # Case: Non-circular Binary

    n = _numpy.sqrt(G*M12/a**3) # compute the mean motion of the planet
    
    w, W, i = star.omega, star.Omega, star.inc # redefine the orientation elements of the flyby star for convenience
    V = star.v
    # GM123 = G*M123 
    q = (- mu + _numpy.sqrt( mu**2. + star.b**2. * V**4.))/V**2. # compute the periapsis of the flyby star

    # Calculate the following convenient functions of the planet's eccentricity and Bessel functions of the first kind of order n.
    e1 = _jv(-1,e) - 2*e*_j0(e) + 2*e*_jv(2,0) - _jv(3,e)
    e2 = _jv(-1,e) - _jv(3,e)
    e4 = _jv(-1,e) - e*_j0(e) - e*_jv(2,e) + _jv(3,e)

    # Calculate a convenient function of the planet's semi-major axis and the flyby star's periapsis.
    k = _numpy.sqrt((2*M12*q**3)/(M123*a**3))
    
    # Calculate convenient functions of the flyby star's eccentricity.
    f1 = ((es + 1.0)**(0.75)) / ((2.0**(0.75)) * (es*es))
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        f2 = (3.0/(2.0*_numpy.sqrt(2.0))) * (_numpy.sqrt((es*es) - 1.0) - _numpy.arccos(1.0/es)) / ((es - 1.0)**(1.5))
    
    # Compute the prefactor and terms of the calculation done by Roy & Haddow (2003)
    prefactor = (-_numpy.sqrt(_numpy.pi)/8.0) * ((G*m1*m2*m3)/M12) * ((a*a)/(q*q*q)) * f1 * k**(2.5) * _numpy.exp((-2.0*k/3.0)*f2)
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        term1 = e1 * ( _numpy.sin(2.0*w + n*t0)*_numpy.cos(2.0*i - 1.0)- _numpy.sin(2.0*w + n*t0)*_numpy.cos(2.0*i)*_numpy.cos(2.0*W) - 3.0*_numpy.sin(n*t0 + 2.0*w)*_numpy.cos(2.0*W) - 4.0*_numpy.sin(2.0*W)*_numpy.cos(2.0*w + n*t0)*_numpy.cos(i) )
        term2 = e2 * (1.0 - e*e) * ( _numpy.sin(2.0*w + n*t0)*(1.0-_numpy.cos(2.0*i)) - _numpy.sin(2.0*w + n*t0)*_numpy.cos(2.0*i)*_numpy.cos(2.0*W) - 3.0*_numpy.sin(n*t0 +2.0*w)*_numpy.cos(2.0*W) - 4.0*_numpy.cos(n*t0 + 2.0*w)*_numpy.sin(2.0*W)*_numpy.cos(i) )
        term3 = e4 * _numpy.sqrt(1.0 - e*e) * (-2.0*_numpy.cos(2.0*i)*_numpy.cos(2.0*w + n*t0)*_numpy.sin(2.0*W) - 6.0*_numpy.cos(2.0*w + n*t0)*_numpy.sin(2.0*W) - 8.0*_numpy.cos(2.0*W)*_numpy.sin(2.0*w + n*t0)*_numpy.cos(i) )
    
    noncircular_result = None
    if averaged: noncircular_result = (prefactor * (e1 + e2 * (1 - e*e) + 2 * e4 * _numpy.sqrt(1 - e*e))).decompose(list(unit_set.values()))
    else: noncircular_result = (prefactor * ( term1 + term2 + term3)).decompose(list(unit_set.values()))

    # Case: Circular Binary

    # Compute the prefactor and terms of the calculation done by Roy & Haddow (2003)
    prefactor = (-_numpy.sqrt(_numpy.pi)/8.0) * ((G*m1*m2*m3)/M12) * ((a*a*a)/(q*q*q*q)) * f1 * k**(3.5) * _numpy.exp((-2.0*k/3.0)*f2) * (m2/M12 - m1/M12)
    # if m2 < m1: prefactor /= twopi
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        term1 = (1.0 + _numpy.cos(i)) * _numpy.sin(i)**2.0
        term2 = ( (_numpy.cos(w)**3.0) - 3.0 * (_numpy.sin(w)**2.0) * _numpy.cos(w) ) * _numpy.sin(n*t0)
        term3 = ( 3.0 * (_numpy.cos(w)**2.0) * _numpy.sin(w) - (_numpy.sin(w)**3.0)) * _numpy.cos(n*t0)
    
    circular_result = None
    if averaged: circular_result = (prefactor / _numpy.pi).decompose(list(unit_set.values()))
    else: circular_result = (prefactor * term1 * (term2 + term3)).decompose(list(unit_set.values()))

    return circular_result + noncircular_result


def relative_energy_change(sim, star, averaged=False, particle_index=1):
    '''
        An analytical estimate for the relative change in energy of a binary system due to a flyby star.
        
        Combines energy_change_adiabatic_estimate(...) and binary_energy(...) functions.
        
        Parameters
        ----------
        sim : REBOUND Simulation with two bodies, a central star and a planet
        star : AIRBALL Star flyby object
    '''
    return energy_change_adiabatic_estimate(sim=sim, star=star, averaged=averaged, particle_index=particle_index)/binary_energy(sim, particle_index=particle_index)

def parallel_relative_energy_change(sims, stars, averaged=False, particle_index=1):
    '''
        An analytical estimate for the relative change in energy of a binary system due to a flyby star.
        
        Combines energy_change_adiabatic_estimate(...) and binary_energy(...) functions.
        
        Parameters
        ----------
        sims : REBOUND Simulations with two bodies, a central star and a planet
        stars : AIRBALL Stars flyby object
    '''
    return _joblib.Parallel(n_jobs=-1)(_joblib.delayed(relative_energy_change)(sim=sims[i], star=stars[i], averaged=averaged, particle_index=particle_index) for i in range(stars.N))


def eccentricity_change_adiabatic_estimate(sim, star, averaged=False, particle_index=1):
    '''
        An analytical estimate for the change in eccentricity of an eccentric binary system due to a flyby star.
        
        From Equation (7) of Heggie & Rasio (1996) Equation (A3) from Spurzem et al. (2009) https://ui.adsabs.harvard.edu/abs/2009ApJ...697..458S/abstract. 
        The orbital element angles of the flyby star are determined with respect to the plane defined by the binary orbit (the invariant plane). In REBOUND this is the same as when the inclination of the planet is zero.
        
        Parameters
        ----------
        sim : REBOUND Simulation with two bodies, a central star and a planet
        star : AIRBALL Star flyby object
    '''

    index = int(particle_index)

    unit_set = rebound_units(sim)
    t0 = 0*unit_set['time']
    G = (sim.G * unit_set['length']**3 / unit_set['mass'] / unit_set['time']**2)
    
    p = sim.particles
    m1, m2, m3 = p[0].m * unit_set['mass'], p[index].m * unit_set['mass'], star.mass # redefine the masses for convenience
    M12 = m1 + m2 # total mass of the binary system
    M123 = m1 + m2 + m3 # total mass of all the objects involved
    
    mu = G * (system_mass(sim)  * unit_set['mass'] + m3)
    es = vinf_and_b_to_e(mu=mu, star_b=star.b, star_v=star.v)
    
    a, e = p[index].a * unit_set['length'], p[index].e # redefine the orbital elements of the planet for convenience
    # n = _numpy.sqrt(G*M12/a**3) # compute the mean motion of the planet
    
    w, W, i = star.omega, star.Omega, star.inc # redefine the orientation elements of the flyby star for convenience
    V = star.v
    # GM123 = G*M123 
    q = (- mu + _numpy.sqrt( mu**2. + star.b**2. * V**4.))/V**2. # compute the periapsis of the flyby star

    prefactor = (-15.0/4.0) * m3 / _numpy.sqrt(M12*M123) * ((a/q)**1.5) * ((e * _numpy.sqrt(1.0 - e*e))/((1.0 + es)**1.5))
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        t1 = (_numpy.sin(i) * _numpy.sin(i) * _numpy.sin(2.0*W) * ( _numpy.arccos(-1.0/es) + _numpy.sqrt(es*es - 1.0) )).value
        t2 = ((1.0/3.0) * (1.0 + _numpy.cos(i)*_numpy.cos(i)) * _numpy.cos(2.0*w) * _numpy.sin(2.0*W)).value
        t3 = (2.0 * _numpy.cos(i) * _numpy.sin(2.0*w) * _numpy.cos(2.0*W) * ((es*es - 1.0)**1.5)/(es*es)).value

    if averaged: return (prefactor * ( ( _numpy.arccos(-1.0/es).value + _numpy.sqrt(es*es - 1.0) ) + (2.0/3.0) + (2.0 * ((es*es - 1.0)**1.5)/(es*es)) )).decompose(list(unit_set.values()))
    else: return (prefactor * (t1 + t2 + t3)).decompose(list(unit_set.values()))

def parallel_eccentricity_change_adiabatic_estimate(sims, stars, averaged=False, particle_index=1):
    '''
        An analytical estimate for the relative change in energy of a binary system due to a flyby star.
        
        Combines energy_change_adiabatic_estimate(...) and binary_energy(...) functions.
        
        Parameters
        ----------
        sims : REBOUND Simulations with two bodies, a central star and a planet
        stars : AIRBALL Stars flyby object
    '''
    return _joblib.Parallel(n_jobs=-1)(_joblib.delayed(eccentricity_change_adiabatic_estimate)(sim=sims[i], star=stars[i], averaged=averaged, particle_index=particle_index) for i in range(stars.N))

def energy_change_close_encounters_sim(sim):
    '''
        An analytical estimate for the change in energy of a binary system due to a flyby star.
        
        From the conclusions of Roy & Haddow (2003) https://ui.adsabs.harvard.edu/abs/2003CeMDA..87..411R/abstract and Heggie (2006) https://ui.adsabs.harvard.edu/abs/2006fbp..book...20H/abstract. The orbital element angles of the flyby star are determined with respect to the plane defined by the binary orbit. In REBOUND this is the same as when the inclination of the planet is zero.
        
        Parameters
        ----------
        sim : three-body REBOUND sim
    '''
    s = sim.copy()
    s.move_to_hel()
    p = s.particles

    G = s.G # Newton's Gravitational constant in units of Msun, AU, Yr2Pi
    m1, m2, m3 = p[0].m, p[1].m, p[2].m # redefine the masses for convenience
    M12 = m1 + m2 # total mass of the binary system
    M23 = m2 + m3 # total mass of the second and third bodies
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
