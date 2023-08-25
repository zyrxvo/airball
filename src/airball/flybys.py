import rebound as _rebound
import numpy as _numpy
import joblib as _joblib

from . import units as u
from .tools import *

############################################################
#################### Flyby Functions #######################
############################################################

def add_star_to_sim(sim, star, rmax=4e5, plane=None):
    
    # Extract the star's characteristics.
    m, b, v, inc, omega, Omega = star.params

    ##################################################
    ## Calculation of Flyby Star Initial Conditions ## 
    ##################################################

    # Calculate the orbital elements of the flyby star.
    e = determine_eccentricity(sim, m, b, star_v=v, star_e=None) # TODO: Allow user to define e instead of v.
    a = -b/_numpy.sqrt(e**2. - 1.) # Compute the semi-major axis of the flyby star
    l = -a*(e*e-1.) # Compute the semi-latus rectum of the hyperbolic orbit (-a because the semi-major axis is negative)

    rmax = verify_unit(rmax, u.au)
    f = _numpy.arccos((l/rmax-1.)/e) # Compute the true anomaly

    #################################################
    
    # Add the flyby star to the simulation.
    sim.move_to_hel() # Move the system into the heliocentric frame of reference.
    if plane is not None:
        int_types = int, _numpy.integer
        # Move the system into the chosen plane of reference.
        rot = _rebound.Rotation.to_new_axes(newz=[0,0,1])
        if plane == 'invariable': rot = _rebound.Rotation.to_new_axes(newz=sim.angular_momentum())
        elif plane == 'ecliptic': rot = _rebound.Rotation.to_new_axes(newz=calculate_angular_momentum(sim)[3]) # Assumes Earth is particle 3. 0-Sun, 1-Mecury, 2-Venus, 3-Earth, ...
        elif isinstance(plane, int_types): rot = _rebound.Rotation.to_new_axes(newz=calculate_angular_momentum(sim)[int(plane)])
        sim.rotate(rot)

    sim.add(m=m.value, a=a.value, e=e.value, f=-f.value, omega=omega.value, Omega=Omega.value, inc=inc.value, hash='flybystar')
    if sim.integrator == 'whfast': sim.ri_whfast.recalculate_coordinates_this_timestep = 1 # Because a new particle was added, we need to tell REBOUND to recalculate the coordinates.
    sim.integrator_synchronize()
    
    if plane is not None: sim.rotate(rot.inverse())
    sim.move_to_com()

    # Because REBOUND Simulations are C structs underneath Python, this function passes the simulation by reference.
    return m,a,e,l

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
def hybrid_flyby(sim, star, rmax=4e5, crossOverFactor=30, overwrite=False, integrator='whckl', heartbeat=None, particle_index=1, plane=None):
    '''
        Simulate a stellar flyby to a REBOUND simulation.
        
        Because REBOUND Simulations are C structs underneath the Python, this function can pass the simulation by reference.
        This can be done by specifying overwrite=True. Meaning, any changes made inside this function to the REBOUND simulation are permanent.
        This function assumes that you are using a WHFAST integrator with REBOUND.
        Uses IAS15 (instead of WHFast) for the closest approach if b < planet_a * crossOverFactor
        
        Parameters
        ----------
        sim : the REBOUND Simulation (star and planets) that will experience the flyby star
        star: a AIRBALL Star object

        rmax : the starting distance of the flyby star in units of AU
        crossOverFactor: the value for when to switch integrators if hybrid=True
        overwrite: determines whether or not to return a copy of sim (overwrite=False) or integrate using the original sim (overwrite=True)
        integrator: sets the integrator for before and after the hybrid switch (for example, if you want to use WHCKL instead of WHFast)
        heartbeat: sets a heartbeat function
        plane: String/Int. The plane defining the orientation of the star, None, 'invariable', 'ecliptic', or Int. Default is None.
    '''

    # Do not overwrite given sim.
    if not overwrite: sim = sim.copy()
    if heartbeat is not None: sim.heartbeat = heartbeat
    sim_units = rebound_units(sim)

    index = int(particle_index)
    m,a,e,l = add_star_to_sim(sim, star, rmax=rmax, plane=plane)

    tperi = sim.particles['flybystar'].T - sim.t # Compute the time to periapsis for the flyby star from the current time.
    
    # Integrate the flyby. Start at the current time and go to twice the time to periapsis.
    rCrossOver = crossOverFactor * sim.particles[index].a * sim_units['length'] # This is the distance to switch integrators
    q = -a * (e-1)
    if q < rCrossOver:
        f = _numpy.arccos((l/rCrossOver-1.)/e) # Compute the true anomaly

        G = (sim.G * sim_units['length']**3 / sim_units['mass'] / sim_units['time']**2)
        mu = G * (sim.calculate_com().m * sim_units['mass'] + m)

        # Compute the time to periapsis from the switching point (-a because the semi-major axis is negative).
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            E = _numpy.arccosh((_numpy.cos(f)+e)/(1.+e*_numpy.cos(f))) # Compute the eccentric anomaly
            M = e * _numpy.sinh(E)-E # Compute the mean anomaly
        tIAS15 = M/_numpy.sqrt(mu/(-a*a*a)) 

        t_switch = sim.t + tperi - tIAS15.value
        t_switch_back = sim.t + tperi + tIAS15.value
        t_end = sim.t + 2*tperi

        dt = sim.dt
        dt_frac = sim.dt/sim.particles[1].P

        sim.integrate(t_switch)
        sim.ri_whfast.recalculate_coordinates_this_timestep = 1
        sim.integrator_synchronize()

        sim.integrator = 'ias15'
        sim.gravity = 'basic'
        sim.integrate(t_switch_back)
        
        sim.integrator = integrator
        sim.ri_whfast.safe_mode = 0
        sim.ri_whfast.recalculate_coordinates_this_timestep = 1
        sim.integrator_synchronize()
        if sim.particles[1].P > 0: sim.dt = dt_frac*sim.particles[1].P
        else: sim.dt = dt
        
        sim.integrate(t_end)
    else:
        sim.integrate(sim.t + 2*tperi)
        sim.ri_whfast.recalculate_coordinates_this_timestep = 1
        sim.integrator_synchronize()
    
    # Remove the flyby star. 
    sim.remove(hash='flybystar')
    sim.ri_whfast.recalculate_coordinates_this_timestep = 1 # Because a particle was removed, we need to tell REBOUND to recalculate the coordinates and to synchronize.
    sim.integrator_synchronize()
    sim.move_to_com() # Readjust the system back into the centre of mass/momentum frame for integrating.
    
    return sim


def hybrid_flybys(sims, stars, **kwargs):
    '''
        Run serial flybys in parallel.

        Parameters
        ---------------
        sims: REBOUND Simulations. Required.
        stars: AIRBALL Stars. Required.

        integrator: String. The integrator to use for the main integration (IAS15 is used for the close encounter). Default is WHCKL.
        crossOverFactor: Float. The value for when to switch to IAS15 as a multiple of sim.particles[1].a Default is 30.
        overwrite: True/False. Sets whether or not to return new simulation objects or overwrite the given ones. Default is False.
        rmax: Float. The starting distance of the flyby object (in units of the REBOUND Simulation). Default is 4e5.
        plane: String/Int. The plane defining the orientation of the star, None, 'invariable', 'ecliptic', or Int. Default is None.
        particle_index: Int. The simulation particle index to define the crossOverFactor with respect to. Default is 1.

        inds: An array of indices to determine which sims and stars to integrate. Default is all of them.
        n_jobs: Integer. The number of jobs per CPU to run in parallel. Default is -1.
        verbose: Integer. The amount of details to display for the parallel jobs. Default is 0.
    '''
    Nruns = 0
    try: 
        Nruns = len(sims)
        if Nruns != len(stars): raise Exception('Sims and stars are unequal lengths')
    except: 
        Nruns = len(stars)
        sims = [sims.copy() for _ in range(Nruns)]
    
    try: 
        rmax = kwargs['rmax']
        if not isList(rmax): rmax = Nruns * [rmax]
        elif len(rmax) != Nruns: raise Exception('List arguments must be same length.')
    except KeyError: rmax = Nruns * [4e5]

    try: 
        crossOverFactor = kwargs['crossOverFactor']
        if not isList(crossOverFactor): crossOverFactor = Nruns * [crossOverFactor]
        elif len(crossOverFactor) != Nruns: raise Exception('List arguments must be same length.')
    except KeyError: crossOverFactor = Nruns * [30]

    try: integrator = kwargs['integrator']
    except KeyError: integrator = 'whckl'

    try: 
        _ = kwargs['heartbeat']
        raise Exception('Cannot parallelize using heartbeat functions.') 
    except KeyError: pass

    inds = kwargs.get('inds', _numpy.arange(Nruns))
    overwrite = kwargs.get('overwrite', False)
    n_jobs = kwargs.get('n_jobs', -1)
    verbose = kwargs.get('verbose', 0)
    particle_index = kwargs.get('particle_index', 1)
    plane = kwargs.get('plane', None)

    sim_results = _joblib.Parallel(n_jobs=n_jobs, verbose=verbose, require='sharedmem')(
    _joblib.delayed(hybrid_flyby)(
        sim=sims[int(i)], star=stars[i], rmax=rmax[i], crossOverFactor=crossOverFactor[i], overwrite=overwrite, integrator=integrator,  particle_index=particle_index, plane=plane) 
    for i in inds)
    
    return sim_results



def flyby(sim, star, rmax=4e5, overwrite=False, heartbeat=None, plane=None):
    '''
        Simulate a stellar flyby to a REBOUND simulation.
        
        Because REBOUND Simulations are C structs underneath the Python, this function can pass the simulation by reference.
        This can be done by specifying overwrite=True. Meaning, any changes made inside this function to the REBOUND simulation are permanent.
        This function assumes that you are using a WHFAST integrator with REBOUND.
        
        Parameters
        ----------
        sim : the REBOUND Simulation (star and planets) that will experience the flyby star
        star: a airball.Star object
        rmax : the starting distance of the flyby star in units of AU
        overwrite: determines whether or not to return a copy of sim (overwrite=False) or integrate using the original sim (overwrite=True)
        heartbeat: set a heartbeat function
        plane: String/Int. The plane defining the orientation of the star, None, 'invariable', 'ecliptic', or Int. Default is None.
    '''

    # Do not overwrite given sim.
    if not overwrite: sim = sim.copy()
    if heartbeat is not None: sim.heartbeat = heartbeat

    add_star_to_sim(sim, star, rmax=rmax, plane=plane)
    
    tperi = sim.particles['flybystar'].T - sim.t # Compute the time to periapsis for the flyby star from the current time.
    
    sim.integrate(sim.t + 2*tperi)

    sim.remove(hash='flybystar')
    if sim.integrator == 'whfast': sim.ri_whfast.recalculate_coordinates_this_timestep = 1 # Because a new particle was added, we need to tell REBOUND to recalculate 
    sim.integrator_synchronize()
    sim.move_to_com() # Readjust the system back into the centre of mass/momentum frame for integrating.
    
    return sim


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

    try: overwrite = kwargs['overwrite']
    except KeyError: overwrite = False

    try: 
        _ = kwargs['heartbeat']
        raise Exception('Cannot parallelize using heartbeat functions.') 
    except KeyError: pass

    try: n_jobs = kwargs['n_jobs']
    except KeyError: n_jobs = -1

    try: verbose = kwargs['verbose']
    except KeyError: verbose = 0

    sim_results = _joblib.Parallel(n_jobs=n_jobs, verbose=verbose, require='sharedmem')(
    _joblib.delayed(flyby)(
        sim=sims[int(i)], star=stars[i], rmax=rmax[i], overwrite=overwrite) 
    for i in inds)
    
    return sim_results

