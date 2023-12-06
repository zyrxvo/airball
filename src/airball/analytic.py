import numpy as _np
import joblib as _joblib
import rebound as _rebound
from scipy.special import j0 as _j0,jv as _jv

from . import tools as _tools
from . import units as _u
from .core import add_star_to_sim

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
    unit_set = _tools.rebound_units(sim)
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
    unit_set = _tools.rebound_units(sim)
    t0 = 0*unit_set['time']
    G = (sim.G * unit_set['length']**3 / unit_set['mass'] / unit_set['time']**2)

    sim = sim.copy()
    sim.rotate(_rebound.Rotation.to_new_axes(newz=sim.angular_momentum()))
    
    p = sim.particles
    m1, m2, m3 = p[0].m * unit_set['mass'], p[index].m * unit_set['mass'], star.mass # redefine the masses for convenience
    M12 = m1 + m2 # total mass of the binary system
    M123 = m1 + m2 + m3 # total mass of all the objects involved
    
    mu = G * (_tools.system_mass(sim)  * unit_set['mass'] + m3)
    es = _tools.vinf_and_b_to_e(mu=mu, star_b=star.b, star_v=star.v)
    
    a, e = p[index].a * unit_set['length'], p[index].e # redefine the orbital elements of the planet for convenience
    # Case: Non-circular Binary

    n = _np.sqrt(G*M12/a**3) # compute the mean motion of the planet
    
    w, W, i = star.omega, star.Omega, star.inc # redefine the orientation elements of the flyby star for convenience
    V = star.v
    # GM123 = G*M123 
    q = (- mu + _np.sqrt( mu**2. + star.b**2. * V**4.))/V**2. # compute the periapsis of the flyby star

    # Calculate the following convenient functions of the planet's eccentricity and Bessel functions of the first kind of order n.
    e1 = _jv(-1,e) - 2*e*_j0(e) + 2*e*_jv(2,0) - _jv(3,e)
    e2 = _jv(-1,e) - _jv(3,e)
    e4 = _jv(-1,e) - e*_j0(e) - e*_jv(2,e) + _jv(3,e)

    # Calculate a convenient function of the planet's semi-major axis and the flyby star's periapsis.
    k = _np.sqrt((2*M12*q**3)/(M123*a**3))
    
    # Calculate convenient functions of the flyby star's eccentricity.
    f1 = ((es + 1.0)**(0.75)) / ((2.0**(0.75)) * (es*es))
    with _u.set_enabled_equivalencies(_u.dimensionless_angles()):
        f2 = (3.0/(2.0*_np.sqrt(2.0))) * (_np.sqrt((es*es) - 1.0) - _np.arccos(1.0/es)) / ((es - 1.0)**(1.5))
    
    # Compute the prefactor and terms of the calculation done by Roy & Haddow (2003)
    prefactor = (-_np.sqrt(_np.pi)/8.0) * ((G*m1*m2*m3)/M12) * ((a*a)/(q*q*q)) * f1 * k**(2.5) * _np.exp((-2.0*k/3.0)*f2)
    with _u.set_enabled_equivalencies(_u.dimensionless_angles()):
        term1 = e1 * ( _np.sin(2.0*w + n*t0)*_np.cos(2.0*i - 1.0)- _np.sin(2.0*w + n*t0)*_np.cos(2.0*i)*_np.cos(2.0*W) - 3.0*_np.sin(n*t0 + 2.0*w)*_np.cos(2.0*W) - 4.0*_np.sin(2.0*W)*_np.cos(2.0*w + n*t0)*_np.cos(i) )
        term2 = e2 * (1.0 - e*e) * ( _np.sin(2.0*w + n*t0)*(1.0-_np.cos(2.0*i)) - _np.sin(2.0*w + n*t0)*_np.cos(2.0*i)*_np.cos(2.0*W) - 3.0*_np.sin(n*t0 +2.0*w)*_np.cos(2.0*W) - 4.0*_np.cos(n*t0 + 2.0*w)*_np.sin(2.0*W)*_np.cos(i) )
        term3 = e4 * _np.sqrt(1.0 - e*e) * (-2.0*_np.cos(2.0*i)*_np.cos(2.0*w + n*t0)*_np.sin(2.0*W) - 6.0*_np.cos(2.0*w + n*t0)*_np.sin(2.0*W) - 8.0*_np.cos(2.0*W)*_np.sin(2.0*w + n*t0)*_np.cos(i) )
    
    noncircular_result = None
    if averaged: noncircular_result = (prefactor * (e1 + e2 * (1 - e*e) + 2 * e4 * _np.sqrt(1 - e*e))).decompose(list(unit_set.values()))
    else: noncircular_result = (prefactor * ( term1 + term2 + term3)).decompose(list(unit_set.values()))

    # Case: Circular Binary

    # Compute the prefactor and terms of the calculation done by Roy & Haddow (2003)
    prefactor = (-_np.sqrt(_np.pi)/8.0) * ((G*m1*m2*m3)/M12) * ((a*a*a)/(q*q*q*q)) * f1 * k**(3.5) * _np.exp((-2.0*k/3.0)*f2) * (m2/M12 - m1/M12)
    # if m2 < m1: prefactor /= twopi
    with _u.set_enabled_equivalencies(_u.dimensionless_angles()):
        term1 = (1.0 + _np.cos(i)) * _np.sin(i)**2.0
        term2 = ( (_np.cos(w)**3.0) - 3.0 * (_np.sin(w)**2.0) * _np.cos(w) ) * _np.sin(n*t0)
        term3 = ( 3.0 * (_np.cos(w)**2.0) * _np.sin(w) - (_np.sin(w)**3.0)) * _np.cos(n*t0)
    
    circular_result = None
    if averaged: circular_result = (prefactor / _np.pi).decompose(list(unit_set.values()))
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


def eccentricity_change_adiabatic_estimate(sim, star, averaged=False, particle_index=1, mode='all', rmax=1e5*_u.au):
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

    unit_set = _tools.rebound_units(sim)
    t0 = sim.t * unit_set['time']
    G = (sim.G * unit_set['length']**3 / unit_set['mass'] / unit_set['time']**2)
    
    p = sim.particles
    m1, m2, m3 = p[0].m * unit_set['mass'], p[index].m * unit_set['mass'], star.mass # redefine the masses for convenience
    M12 = m1 + m2 # total mass of the binary system
    M123 = m1 + m2 + m3 # total mass of all the objects involved
    
    mu = G * (_tools.system_mass(sim)  * unit_set['mass'] + m3)
    es = _tools.vinf_and_b_to_e(mu=mu, star_b=star.b, star_v=star.v)
    
    a, e = p[index].a * unit_set['length'], p[index].e # redefine the orbital elements of the planet for convenience
    # n = _np.sqrt(G*M12/a**3) # compute the mean motion of the planet
    
    w, W, i = star.omega, star.Omega, star.inc # redefine the orientation elements of the flyby star for convenience

    star_params = _tools.hyperbolic_elements_from_stellar_params(sim, star, rmax)
    tperi = star_params['T']
    Mperi = p[index].M + (p[index].n/unit_set['time']) * (tperi - t0) # get the Mean anomaly when the flyby star is at perihelion
    f = _tools.reb_M_to_f(p[index].e, Mperi.value) << _u.rad # get the true anomaly when the flyby star is at perihelion
    Wp = W + f
    V = star.v
    q = (- mu + _np.sqrt( mu**2. + star.b**2. * V**4.))/V**2. # compute the periapsis of the flyby star

    # Case: Non-Circular Binary

    prefactor = (-15.0/4.0) * m3 / _np.sqrt(M12*M123) * ((a/q)**1.5) * ((e * _np.sqrt(1.0 - e*e))/((1.0 + es)**1.5))
    with _u.set_enabled_equivalencies(_u.dimensionless_angles()):
        t1 = (_np.sin(i) * _np.sin(i) * _np.sin(2.0*W) * ( _np.arccos(-1.0/es) + _np.sqrt(es*es - 1.0) )).value
        t2 = ((1.0/3.0) * (1.0 + _np.cos(i)*_np.cos(i)) * _np.cos(2.0*w) * _np.sin(2.0*W)).value
        t3 = (2.0 * _np.cos(i) * _np.sin(2.0*w) * _np.cos(2.0*W) * ((es*es - 1.0)**1.5)/(es*es)).value

    if averaged: noncircular_result = (prefactor * ( ( _np.arccos(-1.0/es).value + _np.sqrt(es*es - 1.0) ) + (2.0/3.0) + (2.0 * ((es*es - 1.0)**1.5)/(es*es)) )).decompose(list(unit_set.values()))
    else: noncircular_result = (prefactor * (t1 + t2 + t3)).decompose(list(unit_set.values()))

    # Case: Circular Binary

    prefactor = (15.0/8.0) * (m3 * _np.abs(m1-m2) / (M12*M12)) * _np.sqrt(M12/M123) * ((a/q)**2.5) * 1.0 / (es*es*es * ((1.0 + es)**2.5))
    
    def f1(e):
        with _u.set_enabled_equivalencies(_u.dimensionless_angles()):
            return ((e**4.0) * _np.arccos(-1.0/e) + (-2.0 * 9.0*e*e + 8.0*e**4.0) * _np.sqrt(e*e - 1.0) / 15.0).to(_u.dimensionless_unscaled)
    with _u.set_enabled_equivalencies(_u.dimensionless_angles()):
        t1 = (_np.cos(i)**2.0 * _np.sin(w)**2.0) * (f1(es) * (1.0 - 3.75 * _np.sin(i)**2.0) + (2.0/15.0) * (es*es - 1.0)**2.5 * (1.0 - 5.0 * _np.sin(w)**2.0 * _np.sin(i)**2.0))**2.0
        t2 = (_np.cos(w)**2.0) * (f1(es) * (1.0 - 1.25 * _np.sin(i)**2.0) + (2.0/15.0) * (es*es - 1.0)**2.5 * (1.0 - 5.0 * _np.sin(w)**2.0 * _np.sin(i)**2.0))**2.0

# (187 (-1 + e^2)^5)/14400 + (199 f1[e] (4 (-1 + e^2)^(5/2) + 15 f1[e]))/7680
    if averaged: 
        with _u.set_enabled_equivalencies(_u.dimensionless_angles()):
            circular_result = prefactor * _np.sqrt(((187.0/14400.0)*(es*es - 1.0)**5.0) + ((199.0 * f1(es)) * (4.0 * (es*es - 1.0)**2.5 + 15.0 * f1(es)))/7680.0)
    else: circular_result = prefactor * _np.sqrt(t1 + t2)

    # Case: Exponential

    prefactor = (3.0 * _np.sqrt(2.0 * _np.pi)) * (m3 * (M12**0.25) / (M123**1.25)) * ((q/a)**0.75) * (((es + 1.0)**0.75)/(es*es))    
    with _u.set_enabled_equivalencies(_u.dimensionless_angles()):
        i2 = i/2.0
        exponential = -_np.sqrt(M12/M123) * ((q/a)**1.5) * ((_np.sqrt(es*es - 1.0) - _np.arccos(1.0/es)) / ((es - 1.0)**1.5))
        angles = (_np.cos(i2)**2.0) * _np.sqrt((_np.cos(i2)**4.0) + ((4.0/9.0)*_np.sin(i2)**4.0) + ((4.0/3.0)*(_np.cos(i2)**2.0)*(_np.sin(i2)**2.0)*_np.cos(4.0*w + 2.0*Wp)))

    if averaged: exponential_result = prefactor * _np.exp(exponential)
    else: exponential_result = prefactor * _np.exp(exponential) * angles

    if mode == 'circular': return circular_result.decompose()
    elif mode == 'noncircular': return noncircular_result.decompose()
    elif mode == 'exponential': return exponential_result.decompose()
    else: return (_np.nan_to_num(circular_result) + _np.nan_to_num(noncircular_result) + _np.nan_to_num(exponential_result)).decompose()

def parallel_eccentricity_change_adiabatic_estimate(sims, stars, averaged=False, particle_index=1, mode='all', rmax=1e5*_u.au):
    '''
        An analytical estimate for the relative change in energy of a binary system due to a flyby star.
        
        Combines energy_change_adiabatic_estimate(...) and binary_energy(...) functions.
        
        Parameters
        ----------
        sims : REBOUND Simulations with two bodies, a central star and a planet
        stars : AIRBALL Stars flyby object
    '''
    return _joblib.Parallel(n_jobs=-1)(_joblib.delayed(eccentricity_change_adiabatic_estimate)(sim=sims[i], star=stars[i], averaged=averaged, particle_index=particle_index, mode=mode, rmax=rmax) for i in range(stars.N))

def eccentricity_change_impulsive_estimate(sim, star, particle_index=1):

    index = int(particle_index)

    unit_set = _tools.rebound_units(sim)
    G = (sim.G * unit_set['length']**3 / unit_set['mass'] / unit_set['time']**2)
    sim = sim.copy()
    # sim.rotate(_rebound.Rotation.to_new_axes(newz=sim.angular_momentum()))
    add_star_to_sim(sim, star, hash='flybystar', rmax=0) # Initialize Star at perihelion
    
    p = sim.particles
    m1, m2, m3 = p[0].m * unit_set['mass'], p[index].m * unit_set['mass'], star.mass # redefine the masses for convenience
    M12 = m1 + m2 # total mass of the binary system
    
    a = p[index].a * unit_set['length'] # redefine the orbital elements of the planet for convenience
    V = star.v
    mu = G * (_tools.system_mass(sim)  * unit_set['mass'] + m3)
    q = (- mu + _np.sqrt( mu**2. + star.b**2. * V**4.))/V**2. # compute the periapsis of the flyby star

    theta = _tools.angle_between(p[index].xyz, p['flybystar'].xyz)

    return ((2.0 * _np.sqrt(G/M12) * (m3/V) * _np.sqrt(a*a*a) / (q*q)) * _np.abs(_np.cos(theta)) * _np.sqrt((_np.cos(theta)**2.0) + (4.0 * _np.sin(theta)**2.0))).decompose()

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
    b = -p[2].a * _np.sqrt(es*es - 1.0)
    q = (- GM123 + _np.sqrt( GM123**2. + b**2. * V**4.))/V**2. # compute the periapsis of the flyby star

    x,y,z = p[2].xyz
    vx,vy,vz = p[1].vxyz

    cosϕ = 1.0/_np.sqrt(1.0 + ((q**2.0)*(V**4.0))/(M23**2.0))
    
    prefactor = (-2.0 * m1 * m2 * m3)/(M12 * M23) * V * cosϕ
    t1 = -(x*vx + y*vy + z*vz)
    t2 = (m3 * V * cosϕ)/M23
    
    sim.move_to_hel()
    
    return prefactor * (t1 + t2)
