import rebound as _rebound
import numpy as _numpy
import joblib as _joblib

from .tools import *

############################################################
#################### Flyby Functions #######################
############################################################

def flyby_particle(sim, star_mass=1, star_b=100, star_v=None,  star_e=None, star_omega='uniform', star_Omega='uniform', star_inc='uniform', star_rmax=2.5e5):
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

    #################################################
    ## Calculation of Flyby Star Initial Conditions ## 
    #################################################

    # Calculate the orbital elements of the flyby star.
    rmax = star_rmax # This is the starting distance of the flyby star in AU
    e = determine_eccentricity(sim, star_mass, star_b, star_v=star_v, star_e=star_e)
    a = -star_b/_numpy.sqrt(e**2. - 1.) # Compute the semi-major axis of the flyby star
    l = -a*(e*e-1.) # Compute the semi-latus rectum of the hyperbolic orbit (-a because the semi-major axis is negative)
    f = _numpy.arccos((l/rmax-1.)/e) # Compute the true anomaly

    #################################################

    return _rebound.Particle(sim, m=star_mass, a=a, e=e, f=-f, omega=star_omega, Omega=star_Omega, inc=star_inc, hash='flybystar')



def flyby(sim, star=None, m=0.3, b=1000, v=40,  e=None, omega='uniform', Omega='uniform', inc='uniform', rmax=2.5e5, hybrid=True, crossOverFactor=30, overwrite=False):
    '''
        Simulate a stellar flyby to a REBOUND simulation.
        
        Because REBOUND Simulations are C structs underneath the Python, this function passes the simulation by reference. 
        Any changes made inside this function to the REBOUND simulation are permanent.
        This function assumes that you are using a WHFAST integrator with REBOUND.
        
        Parameters
        ----------
        sim : the REBOUND Simulation (star and planets) that will experience the flyby star
        m : the mass of the flyby star in units of Msun
        b : impact parameter of the flyby star in units of AU
        v : the relative velocity at infinity between the central star and the flyby star (hyperbolic excess velocity) in units of km/s. Only specify v OR e, not both.
        e : the eccentricity of the flyby star (e > 1). Only specify e OR v, not both.
        omega : the argument of periapsis of the flyby star
        Omega : the longitude of the ascending node of the flyby star
        inc : the inclination of the flyby star
        rmax : the starting distance of the flyby star in units of AU
        hybrid: True/False, use IAS15 (instead of WHFast) for the closest approach if b < planet_a * crossOverFactor
        crossOverFactor: the value for when to switch integrators if hybrid=True
    '''

    # Do not overwrite given sim.
    if not overwrite: sim = sim.copy()
    
    # Extract the star's characteristics.
    if star is not None:
        if len(star) > 3: m, b, v, omega, Omega, inc = star
        else: m, b, v = star

    ##################################################
    ## Calculation of Flyby Star Initial Conditions ## 
    ##################################################

    # Calculate the orbital elements of the flyby star.
    e = determine_eccentricity(sim, m, b, star_v=v, star_e=e)
    a = -b/_numpy.sqrt(e**2. - 1.) # Compute the semi-major axis of the flyby star
    l = -a*(e*e-1.) # Compute the semi-latus rectum of the hyperbolic orbit (-a because the semi-major axis is negative)
    f = _numpy.arccos((l/rmax-1.)/e) # Compute the true anomaly

    #################################################
    
    # Add the flyby star to the simulation. 
    sim.move_to_hel() # Move the system into the heliocentric frame of reference.
    sim.add(m=m, a=a, e=e, f=-f, omega=omega, Omega=Omega, inc=inc, hash='flybystar')
    sim.ri_whfast.recalculate_coordinates_this_timestep = 1 # Because a new particle was added, we need to tell REBOUND to recalculate the coordinates.
    sim.move_to_com() # Move the system back into the centre of mass/momentum frame for integrating.

    tperi = sim.particles['flybystar'].T - sim.t # Compute the time to periapsis for the flyby star from the current time.
    
    de = None
    # Integrate the flyby. Start at the current time and go to twice the time to periapsis.
    if hybrid:
        rCrossOver = crossOverFactor*sim.particles[1].a # This is distance to switch integrators
        
        if b < rCrossOver:
            a = -b/_numpy.sqrt(e**2. - 1.) # Compute the semi-major axis of the flyby star
            l = -a*(e*e-1.) # Compute the semi-latus rectum of the hyperbolic orbit (-a because the semi-major axis is negative)
            f = _numpy.arccos((l/rCrossOver-1.)/e) # Compute the true anomaly
            mu = sim.G * _numpy.sum([p_j.m for p_j in sim.particles])

            # Calculate half of the integration time for the flyby star.
            E = _numpy.arccosh((_numpy.cos(f)+e)/(1.+e*_numpy.cos(f))) # Compute the eccentric anomaly
            M = e * _numpy.sinh(E)-E # Compute the mean anomaly
            tIAS15 = M/_numpy.sqrt(mu/(-a*a*a)) # Compute the time to periapsis (-a because the semi-major axis is negative)

            t1 = sim.t + tperi - tIAS15
            t2 = sim.t + tperi
            t3 = sim.t + tperi + tIAS15
            t4 = sim.t + 2*tperi

            dt = sim.dt
            dt_frac = sim.dt/sim.particles[1].P
            # print(f'\n::Initial::\ndt: {sim.dt:6.2f}\na: {sim.particles[1].a:6.2f}\ne: {sim.particles[1].e:6.2f}')

            sim.integrate(t1, exact_finish_time=0)
            sim.ri_whfast.recalculate_coordinates_this_timestep = 1
            sim.integrator_synchronize()

            sim.integrator = 'ias15'
            sim.gravity = 'basic'
            sim.integrate(t2)

            # de = airball.energy_change_close_encounters_sim(sim)
            sim.move_to_com()

            sim.integrate(t3)
            
            sim.integrator = 'whfast'
            sim.ri_whfast.safe_mode = 0
            sim.ri_whfast.recalculate_coordinates_this_timestep = 1
            sim.integrator_synchronize()
            if sim.particles[1].P > 0: sim.dt = dt_frac*sim.particles[1].P
            else: sim.dt = dt
            
            sim.integrate(t4, exact_finish_time=0)
        else:
            t1 = sim.t + tperi
            t2 = sim.t + 2*tperi

            sim.integrate(t1, exact_finish_time=0)
            sim.ri_whfast.recalculate_coordinates_this_timestep = 1
            sim.integrator_synchronize()
            # de = airball.energy_change_close_encounters_sim(sim)
            sim.move_to_com()
            sim.integrate(t2, exact_finish_time=0)
            sim.ri_whfast.recalculate_coordinates_this_timestep = 1
            sim.integrator_synchronize()
    else:
        t1 = sim.t + tperi
        t2 = sim.t + 2*tperi
        sim.integrate(t1, exact_finish_time=0)
        sim.ri_whfast.recalculate_coordinates_this_timestep = 1
        sim.integrator_synchronize()
        # de = airball.energy_change_close_encounters_sim(sim)
        sim.move_to_com()
        sim.integrate(t2, exact_finish_time=0)
        sim.ri_whfast.recalculate_coordinates_this_timestep = 1
        sim.integrator_synchronize()
    
    # Remove the flyby star. 
    sim.remove(hash='flybystar')
    sim.ri_whfast.recalculate_coordinates_this_timestep = 1 # Because a particle was removed, we need to tell REBOUND to recalculate the coordinates and to synchronize.
    sim.integrator_synchronize()
    sim.move_to_com() # Readjust the system back into the centre of mass/momentum frame for integrating.
    
    return sim


def flybys(sims, **kwargs):
    '''
        Run serial flybys in parallel.
    '''
    Nruns = len(sims)

    try: stars = kwargs['stars']
    except: stars = Nruns * [None]
    try:
        m = kwargs['m']
        if not isList(m): m = Nruns * [m]
        else: assert len(m) == Nruns
    except AssertionError: raise Exception('List arguments must be same length.')
    except KeyError: m = Nruns * [None]
    try:
        b = kwargs['b']
        if not isList(b): b = Nruns * [b]
        assert len(b) == Nruns
    except AssertionError: raise Exception('List arguments must be same length.')
    except KeyError: b = Nruns * [None]
    try: 
        v = kwargs['v']
        if not isList(v): v = Nruns * [v]
        else: assert len(v) == Nruns
    except AssertionError: raise Exception('List arguments must be same length.')
    except KeyError: v = Nruns * [None]
    try: 
        e = kwargs['e']
        if not isList(e): e = Nruns * [e]
        else: assert len(e) == Nruns
    except AssertionError: raise Exception('List arguments must be same length.')
    except KeyError: e = Nruns * [None]
    try: 
        omega = kwargs['omega']
        if not isList(omega): omega = Nruns * [omega]
        else: assert len(omega) == Nruns
    except AssertionError: raise Exception('List arguments must be same length.')
    except KeyError: omega = Nruns * ['uniform']
    try: 
        Omega = kwargs['Omega']
        if not isList(Omega): Omega = Nruns * [Omega]
        else: assert len(Omega) == Nruns
    except AssertionError: raise Exception('List arguments must be same length.')
    except KeyError: Omega = Nruns * ['uniform']
    try: 
        inc = kwargs['inc']
        if not isList(inc): inc = Nruns * [inc]
        else: assert len(inc) == Nruns
    except AssertionError: raise Exception('List arguments must be same length.')
    except KeyError: inc = Nruns * ['uniform']
    try: 
        rmax = kwargs['rmax']
        if not isList(rmax): rmax = Nruns * [rmax]
        else: assert len(rmax) == Nruns
    except AssertionError: raise Exception('List arguments must be same length.')
    except KeyError: rmax = Nruns * [2.5e5]
    try: hybrid = kwargs['hybrid']
    except KeyError: hybrid = True
    try: crossOverFactor = kwargs['crossOverFactor']
    except KeyError: crossOverFactor = 30
    try: overwrite = kwargs['overwrite']
    except KeyError: overwrite = False
    try: n_jobs = kwargs['n_jobs']
    except KeyError: n_jobs = -1
    try: verbose = kwargs['verbose']
    except KeyError: verbose = 0

    sim_results = _joblib.Parallel(n_jobs=n_jobs, verbose=verbose, require='sharedmem')(
    _joblib.delayed(flyby)(
        sim=sims[i], star=stars[i], m=m[i], b=b[i], v=v[i], e=e[i], omega=omega[i], Omega=Omega[i], inc=inc[i], rmax=rmax[i], hybrid=hybrid, crossOverFactor=crossOverFactor, overwrite=overwrite) 
    for i in range(Nruns))
    
    return sim_results

