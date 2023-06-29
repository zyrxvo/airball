import rebound as _rebound
import numpy as _numpy
import joblib as _joblib

from . import units
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



# def flyby(sim, star=None, m=0.3, b=1000, v=40,  e=None, inc='uniform', omega='uniform', Omega='uniform', rmax=2.5e5, hybrid=True, crossOverFactor=30, overwrite=False):
def flyby(sim, star, rmax=4e5, hybrid=True, crossOverFactor=30, overwrite=False):
    '''
        Simulate a stellar flyby to a REBOUND simulation.
        
        Because REBOUND Simulations are C structs underneath the Python, this function can pass the simulation by reference.
        This can be done by specifying overwrite=True. Meaning, any changes made inside this function to the REBOUND simulation are permanent.
        This function assumes that you are using a WHFAST integrator with REBOUND.
        
        Parameters
        ----------
        sim : the REBOUND Simulation (star and planets) that will experience the flyby star
        m : the mass of the flyby star in units of Msun
        b : impact parameter of the flyby star in units of AU
        v : the relative velocity at infinity between the central star and the flyby star (hyperbolic excess velocity) in units of km/s. Only specify v OR e, not both.
        e : the eccentricity of the flyby star (e > 1). Only specify e OR v, not both.
        inc : the inclination of the flyby star
        omega : the argument of periapsis of the flyby star
        Omega : the longitude of the ascending node of the flyby star
        rmax : the starting distance of the flyby star in units of AU
        hybrid: True/False, use IAS15 (instead of WHFast) for the closest approach if b < planet_a * crossOverFactor
        crossOverFactor: the value for when to switch integrators if hybrid=True
        overwrite: determines whether or not to return a copy of sim (overwrite=False) or integrate using the original sim (overwrite=True)
    '''

    # Do not overwrite given sim.
    if not overwrite: sim = sim.copy()
    sim_units = rebound_units(sim)
    
    # Extract the star's characteristics.
    m, b, v, omega, Omega, inc = star.params

    ##################################################
    ## Calculation of Flyby Star Initial Conditions ## 
    ##################################################

    # Calculate the orbital elements of the flyby star.
    e = determine_eccentricity(sim, m, b, star_v=v, star_e=None) # TODO: Allow user to define e instead of v.
    a = -b/_numpy.sqrt(e**2. - 1.) # Compute the semi-major axis of the flyby star
    l = -a*(e*e-1.) # Compute the semi-latus rectum of the hyperbolic orbit (-a because the semi-major axis is negative)

    rmax = rmax.to(units.au) if isQuantity(rmax) else rmax * units.au
    f = _numpy.arccos((l/rmax-1.)/e) # Compute the true anomaly

    #################################################
    
    # Add the flyby star to the simulation. 
    rot = _rebound.Rotation.to_new_axes(newz=sim.angular_momentum())
    sim.move_to_hel() # Move the system into the heliocentric, invariable plane frame of reference.
    sim.rotate(rot)
    sim.add(m=m.value, a=a.value, e=e.value, f=-f.value, omega=omega.value, Omega=Omega.value, inc=inc.value, hash='flybystar')
    sim.ri_whfast.recalculate_coordinates_this_timestep = 1 # Because a new particle was added, we need to tell REBOUND to recalculate the coordinates.
    sim.rotate(rot.inverse())
    sim.move_to_com() # Move the system back into the centre of mass, original frame for integrating.

    tperi = sim.particles['flybystar'].T - sim.t # Compute the time to periapsis for the flyby star from the current time.
    
    # Try to integrate the flyby. Start at the current time and go to twice the time to periapsis.
    # try:
    if hybrid:
        rCrossOver = crossOverFactor * sim.particles[1].a * sim_units['length'] # This is distance to switch integrators
        
        if b < rCrossOver:
            a = -b/_numpy.sqrt(e**2. - 1.) # Compute the semi-major axis of the flyby star
            l = -a*(e*e-1.) # Compute the semi-latus rectum of the hyperbolic orbit (-a because the semi-major axis is negative)
            f = _numpy.arccos((l/rCrossOver-1.)/e) # Compute the true anomaly

            G = (sim.G * sim_units['length']**3 / sim_units['mass'] / sim_units['time']**2)
            mu = G * (_numpy.sum([p.m for p in sim.particles]) * sim_units['mass'] + m)

            # Calculate half of the integration time for the flyby star.
            with units.set_enabled_equivalencies(units.dimensionless_angles()):
                E = _numpy.arccosh((_numpy.cos(f)+e)/(1.+e*_numpy.cos(f))) # Compute the eccentric anomaly
                M = e * _numpy.sinh(E)-E # Compute the mean anomaly

            tIAS15 = M/_numpy.sqrt(mu/(-a*a*a)) # Compute the time to periapsis (-a because the semi-major axis is negative)

            t1 = sim.t + tperi - tIAS15.value
            t2 = sim.t + tperi
            t3 = sim.t + tperi + tIAS15.value
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
    # except Exception as err:
    #     print(err)
    #     return sim


def flybys(sims, stars, **kwargs):
    '''
        Run serial flybys in parallel.
    '''
    Nruns = 0
    try: 
        Nruns = len(sims)
        if Nruns != len(stars): raise Exception('Sims and stars are unequal lengths')
    except: 
        Nruns = len(stars)
        sims = [sims.copy() for _ in range(Nruns)]
    
    try: inds = kwargs['inds']
    except KeyError: inds = _numpy.arange(Nruns)

    try: 
        rmax = kwargs['rmax']
        if not isList(rmax): rmax = Nruns * [rmax]
        elif len(rmax) != Nruns: raise Exception('List arguments must be same length.')
    except KeyError: rmax = Nruns * [4e5]

    try: 
        hybrid = kwargs['hybrid']
        if not isList(hybrid): hybrid = Nruns * [hybrid]
        elif len(hybrid) != Nruns: raise Exception('List arguments must be same length.')
    except KeyError: hybrid = Nruns * [True]

    try: 
        crossOverFactor = kwargs['crossOverFactor']
        if not isList(crossOverFactor): crossOverFactor = Nruns * [crossOverFactor]
        elif len(crossOverFactor) != Nruns: raise Exception('List arguments must be same length.')
    except KeyError: crossOverFactor = Nruns * [30]

    try: overwrite = kwargs['overwrite']
    except KeyError: overwrite = False

    try: n_jobs = kwargs['n_jobs']
    except KeyError: n_jobs = -1

    try: verbose = kwargs['verbose']
    except KeyError: verbose = 0

    sim_results = _joblib.Parallel(n_jobs=n_jobs, verbose=verbose, require='sharedmem')(
    _joblib.delayed(flyby)(
        sim=sims[int(i)], star=stars[i], rmax=rmax[i], hybrid=hybrid[i], crossOverFactor=crossOverFactor[i], overwrite=overwrite) 
    for i in inds)
    
    return sim_results

