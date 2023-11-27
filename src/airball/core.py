import rebound as _rebound
import numpy as _numpy
import joblib as _joblib
import warnings as _warnings
import tempfile as _tempfile

from . import tools as _tools
from . import units as _u

############################################################
################# Flyby Helper Functions ###################
############################################################

def _rotate_into_plane(sim, plane):
    '''
        Rotates the simulation into the specified plane.
    '''
    int_types = int, _numpy.integer
    rotation = _rebound.Rotation.to_new_axes(newz=[0,0,1])
    if plane is not None:
        # Move the system into the chosen plane of reference. TODO: Make sure the angular momentum calculations don't include other flyby stars.
        if plane == 'invariable': rotation = _rebound.Rotation.to_new_axes(newz=sim.angular_momentum())
        elif plane == 'ecliptic': rotation = _rebound.Rotation.to_new_axes(newz=_tools.calculate_angular_momentum(sim)[3]) # Assumes Earth is particle 3. 0-Sun, 1-Mecury, 2-Venus, 3-Earth, ...
        elif isinstance(plane, int_types): rotation = _rebound.Rotation.to_new_axes(newz=_tools.calculate_angular_momentum(sim)[int(plane)])
    sim.rotate(rotation)
    return rotation

def add_star_to_sim(sim, star, hash, **kwargs):
    # Because REBOUND Simulations are C structs underneath Python, this function passes the simulation by reference.

    units = _tools.rebound_units(sim)
    rmax = _tools.verify_unit(kwargs.get('rmax', 1e5*_u.au), units['length'])
    stellar_elements, semilatus_rectum = _tools.initial_conditions_from_stellar_params(sim, star, rmax)

    plane = kwargs.get('plane')
    if plane is not None: rotation = _rotate_into_plane(sim, plane)

    sim.add(**stellar_elements, hash=hash, primary=sim.particles[0])
    # Because a new particle was added, we need to tell REBOUND to recalculate the coordinates if WHFast is being used.
    if sim.integrator == 'whfast': sim.ri_whfast.recalculate_coordinates_this_timestep = 1
    sim.integrator_synchronize() # For good measure.

    if plane is not None: sim.rotate(rotation.inverse())
    sim.move_to_com()

    # Because REBOUND Simulations are C structs underneath Python, this function passes the simulation by reference.
    return {'m':stellar_elements['m'] * units['mass'], 'a':stellar_elements['a'] * units['length'], 'e':stellar_elements['e'], 'l':semilatus_rectum * units['length']}

def remove_star_from_sim(sim, hash):
    # Because REBOUND Simulations are C structs underneath Python, this function passes the simulation by reference.
    sim.remove(hash=hash)
    # Because a particle was removed, we need to tell REBOUND to recalculate the coordinates if WHFast is being used and to synchronize.
    if sim.integrator == 'whfast': sim.ri_whfast.recalculate_coordinates_this_timestep = 1
    sim.integrator_synchronize()
    sim.move_to_com() # Readjust the system back into the centre of mass/momentum frame for integrating.
    # Because REBOUND Simulations are C structs underneath Python, this function passes the simulation by reference.

def _time_to_periapsis_from_crossover_point(sim, sim_units, crossoverFactor, index, star_elements):
    '''
        Compute the time to periapsis from crossover point.
    '''
    rCrossOver = crossoverFactor * sim.particles[index].a * sim_units['length'] # This is the distance to switch integrators
    q = star_elements['a'] * (1 - star_elements['e'])
    if q < rCrossOver:
        f = _numpy.arccos((star_elements['l']/rCrossOver-1.)/star_elements['e']) # Compute the true anomaly for the cross-over point.

        G = (sim.G * sim_units['length']**3 / sim_units['mass'] / sim_units['time']**2)
        mu = G * (_tools.system_mass(sim) * sim_units['mass'] + star_elements['m'])

        # Compute the time to periapsis from the switching point (-a because the semi-major axis is negative).
        with _u.set_enabled_equivalencies(_u.dimensionless_angles()):
            E = _numpy.arccosh((_numpy.cos(f)+star_elements['e'])/(1.+star_elements['e']*_numpy.cos(f))) # Compute the eccentric anomaly
            M = star_elements['e'] * _numpy.sinh(E)-E # Compute the mean anomaly
        return True, M/_numpy.sqrt(mu/(-star_elements['a']*star_elements['a']*star_elements['a']))
    else: return False, None

# old signature flyby(sim, star=None, m=0.3, b=1000, v=40,  e=None, inc='uniform', omega='uniform', Omega='uniform', rmax=2.5e5, hybrid=True, crossoverFactor=30, overwrite=False):

def _integrate_with_ias15(sim, tmax):
    sim.integrator = 'ias15'
    sim.gravity = 'basic'
    sim.integrate(tmax)

def _integrate_with_whckl(sim, tmax, dt, dt_frac):
    sim.integrator = 'whckl'
    sim.ri_whfast.safe_mode = 0
    sim.ri_whfast.recalculate_coordinates_this_timestep = 1
    sim.integrator_synchronize()
    if sim.particles[1].P > 0: sim.dt = dt_frac*sim.particles[1].P
    else: sim.dt = dt
    sim.integrate(tmax)
    sim.ri_whfast.recalculate_coordinates_this_timestep = 1
    sim.integrator_synchronize()

############################################################
#################### Flyby Functions #######################
############################################################

def hybrid_flyby(sim, star, **kwargs):
    '''
        Simulate a stellar flyby to a REBOUND simulation.

        Because REBOUND Simulations are C structs underneath the Python, this function can pass the simulation by reference.
        Meaning, any changes made inside this function to the REBOUND simulation are permanent. This can be avoided by specifying overwrite=False.
        This function assumes that you are using the WHCKL integrator with REBOUND.
        Uses IAS15 (instead of WHCKL) for the closest approach if b < planet_a * crossoverFactor

        Parameters
        ----------
        sim : the REBOUND Simulation (star and planets) that will experience the flyby star
        star: a AIRBALL Star object

        rmax : the starting distance of the flyby star in units of AU
        crossoverFactor: the value for when to switch integrators, i.e. 30 times the semi-major axis of particle 1. Default is 30x.
        particle_index: the particle index to consider for the crossoverFactor. Default is 1.
        overwrite: determines whether or not to return a copy of sim (overwrite=False) or integrate using the original sim (overwrite=True)
        integrator: sets the integrator for before and after the hybrid switch (for example, if you want to use WHCKL instead of WHFast)
        plane: String/Int. The plane defining the orientation of the star, None, 'invariable', 'ecliptic', or Int. Default is None.
    '''

    overwrite = kwargs.get('overwrite', True)
    if not overwrite: sim = sim.copy()
    hash = kwargs.get('hash', 'flybystar')
    sim_units = _tools.rebound_units(sim)

    star_vars = add_star_to_sim(sim, star, rmax=kwargs.get('rmax', 1e5*_u.au), plane=kwargs.get('plane'), hash=hash)

    tperi = sim.particles[hash].T - sim.t # Compute the time to periapsis for the flyby star from the current time.

    # Integrate the flyby. Start at the current time and go to twice the time to periapsis.
    switch, tIAS15 = _time_to_periapsis_from_crossover_point(sim, sim_units, crossoverFactor=kwargs.get('crossoverFactor', 30), index=kwargs.get('particle_index', 1), star_elements=star_vars)
    if switch:
        t_switch = sim.t + tperi - tIAS15.value
        t_switch_back = sim.t + tperi + tIAS15.value
        t_end = sim.t + 2*tperi

        dt = sim.dt
        dt_frac = sim.dt/sim.particles[1].P

        _integrate_with_whckl(sim, t_switch, dt, dt_frac)
        _integrate_with_ias15(sim, t_switch_back)
        _integrate_with_whckl(sim, t_end, dt, dt_frac)

    else: _integrate_with_whckl(sim, tmax=(sim.t + 2*tperi), dt=sim.dt, dt_frac=sim.dt/sim.particles[1].P)

    # Remove the flyby star.
    remove_star_from_sim(sim, hash=hash)

    return sim

def hybrid_flybys(sims, stars, **kwargs):
    '''
        Run serial flybys in parallel.

        Parameters
        ---------------
        sims : A list of REBOUND Simulations.
            REBOUND simulations to integrate flybys with. If only one simulation is given, then AIRBALL will duplicate it to match the number of Stars given. Required.
        stars : AIRBALL Stars.
            The objects that will flyby the given REBOUND simulations. Required.

        crossoverFactor : Float.
            The value for when to switch to IAS15 as a multiple of sim.particles[1].a Default is 30.
        overwrite : True/False.
            Sets whether or not to return new simulation objects or overwrite the given ones. Default is True, meaning the same simulation object will be returned. This keeps all original pointers attached to it.
        rmax : Float.
            The starting distance of the flyby object (in units of the REBOUND Simulation). Default is 1e5.
        plane : String/Int.
            The plane defining the orientation of the star, None, 'invariable', 'ecliptic', or Int. Default is None.
        particle_index : Int.
            The simulation particle index to define the crossoverFactor with respect to. Default is 1.
        hashes : array_like.
            A list of hash values for adding and removing stars from simulations. Default is 'flybystar'.

        inds : array_like
            An array of indices to determine which sims and stars to integrate. Default is all of them.
        n_jobs : Integer.
            The number of jobs per CPU to run in parallel. Default is -1.
        verbose : Integer.
            The amount of details to display for the parallel jobs. Default is 0.

        Returns
        -------------
        hybrid_flybys : list
            List of REBOUND simulations that experienced a flyby.
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
        if not _tools.isList(rmax): rmax = Nruns * [rmax]
        elif len(rmax) != Nruns: raise Exception('List arguments must be same length.')
    except KeyError: rmax = Nruns * [1e5]

    try:
        crossoverFactor = kwargs['crossoverFactor']
        if not _tools.isList(crossoverFactor): crossoverFactor = Nruns * [crossoverFactor]
        elif len(crossoverFactor) != Nruns: raise Exception('List arguments must be same length.')
    except KeyError: crossoverFactor = Nruns * [30]

    try:
        hashes = kwargs['hashes']
        if not _tools.isList(hashes): hashes = Nruns * [hashes]
        elif len(hashes) != Nruns: raise Exception('List arguments must be same length.')
    except KeyError: hashes = Nruns * ['flybystar']

    inds = kwargs.get('inds', _numpy.arange(Nruns))
    overwrite = kwargs.get('overwrite', True)
    n_jobs = kwargs.get('n_jobs', -1)
    verbose = kwargs.get('verbose', 0)
    particle_index = kwargs.get('particle_index', 1)
    plane = kwargs.get('plane', None)

    sim_results = _joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
    _joblib.delayed(hybrid_flyby)(
        sim=sims[int(i)], star=stars[i], rmax=rmax[i], crossoverFactor=crossoverFactor[i], overwrite=overwrite, particle_index=particle_index, plane=plane, hash=hashes[i])
    for i in inds)

    return sim_results

def flyby(sim, star, **kwargs):
    '''
        Simulate a stellar flyby to a REBOUND simulation.

        Because REBOUND Simulations are C structs underneath the Python, this function can pass the simulation by reference.
        Meaning, any changes made inside this function to the REBOUND simulation are permanent. This can be avoided by specifying overwrite=False.
        This function assumes that you are using a WHFAST integrator with REBOUND.

        Parameters
        ----------
        sim : the REBOUND Simulation (star and planets) that will experience the flyby star
        star: a airball.Star object

        rmax : the starting distance of the flyby star in units of AU
        overwrite: determines whether or not to return a copy of sim (overwrite=False) or integrate using the original sim (overwrite=True)
        plane: String/Int. The plane defining the orientation of the star, None, 'invariable', 'ecliptic', or Int. Default is None.
        hash: String. The name for the flyby star. Default is `flybystar`.
    '''
    if kwargs.get('hybrid', False): return hybrid_flyby(sim, star, **kwargs)
    else:
        overwrite = kwargs.get('overwrite', True)
        if not overwrite: sim = sim.copy()
        hash = kwargs.get('hash', 'flybystar')
        add_star_to_sim(sim, star, hash, rmax=kwargs.get('rmax', 1e5*_u.au), plane=kwargs.get('plane'))
        tperi = sim.particles[hash].T - sim.t # Compute the time to periapsis for the flyby star from the current time.
        sim.integrate(sim.t + 2*tperi)
        remove_star_from_sim(sim, hash)
        return sim

def flybys(sims, stars, **kwargs):
    '''
        Run serial flybys in parallel.

        Parameters
        ---------------
        sims : A list of REBOUND Simulations.
            REBOUND simulations to integrate flybys with. If only one simulation is given, then AIRBALL will duplicate it to match the number of Stars given. Required.
        stars : AIRBALL Stars.
            The objects that will flyby the given REBOUND simulations. Required.

        overwrite : True/False.
            Sets whether or not to return new simulation objects or overwrite the given ones. Default is True, meaning the same simulation object will be returned. This keeps all original pointers attached to it.
        rmax : Float.
            The starting distance of the flyby object (in units of the REBOUND Simulation). Default is 1e5.
        plane : String/Int.
            The plane defining the orientation of the star, None, 'invariable', 'ecliptic', or Int. Default is None.

        inds : array_like
            An array of indices to determine which sims and stars to integrate. Default is all of them.
        n_jobs : Integer.
            The number of jobs per CPU to run in parallel. Default is -1.
        verbose : Integer.
            The amount of details to display for the parallel jobs. Default is 0.

        Returns
        -------------
        flybys : list
            List of REBOUND simulations that experienced a flyby.
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
        rmax = _tools.verify_unit(rmax, _u.au)
        if len(rmax.shape) == 0: rmax = _numpy.array(stars.N * [rmax.value]) << rmax.unit
        elif len(rmax) != stars.N: raise Exception('List arguments must be same length.')
    except KeyError: rmax = _numpy.array(Nruns * [1e5]) << _u.au
    if _numpy.any(stars.b > rmax): raise Exception('Some stellar impact parameters are greater than the stellar starting distance, rmax.')

    heartbeat = kwargs.get('heartbeat', None)
    inds = kwargs.get('inds', _numpy.arange(Nruns))
    overwrite = kwargs.get('overwrite', True)
    n_jobs = kwargs.get('n_jobs', -1)
    verbose = kwargs.get('verbose', 0)
    plane = kwargs.get('plane', None)

    sim_results = _joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
    _joblib.delayed(flyby)(
        sim=sims[int(i)], star=stars[i], rmax=rmax[i], overwrite=overwrite, heartbeat=heartbeat, plane=plane)
    for i in inds)

    return sim_results

def successive_flybys(sim, stars, **kwargs):
    '''
        Simulate a stellar flyby to a REBOUND simulation.

        Because REBOUND Simulations are C structs underneath the Python, this function can pass the simulation by reference.
        Meaning, any changes made inside this function to the REBOUND simulation are permanent. This can be avoided by specifying overwrite=False.
        This function assumes that you are using a WHFAST integrator with REBOUND.
        Uses IAS15 (instead of WHFast) for the closest approach if b < planet_a * crossoverFactor

        Parameters
        ----------
        sim : the REBOUND Simulation (star and planets) that will experience the flyby star
        star: a AIRBALL Star object

        rmax : the starting distance of the flyby star in units of AU
        crossoverFactor: the value for when to switch integrators if hybrid=True
        overwrite: determines whether or not to return a copy of sim (overwrite=False) or integrate using the original sim (overwrite=True)
        integrator: sets the integrator for before and after the hybrid switch (for example, if you want to use WHCKL instead of WHFast)
        heartbeat: sets a heartbeat function
        plane: String/Int. The plane defining the orientation of the star, None, 'invariable', 'ecliptic', or Int. Default is None.
    '''

    # Do not overwrite given sim.
    overwrite = kwargs.get('overwrite', True)
    if overwrite == False: 
        hb = sim.heartbeat
        sim = sim.copy()
        sim.heartbeat = hb
    hashes = kwargs.get('hashes', [f'flybystar{i}' for i in range(stars.N)])
    for i,star in enumerate(stars):
        if overwrite == True: flyby(sim, star, hash=hashes[i], **kwargs)
        else:  sim = flyby(sim, star, hash=hashes[i], **kwargs)
    return sim

def concurrent_flybys(sim, stars, start_times, **kwargs):
    '''
        Simulate a stellar flyby to a REBOUND simulation.

        Because REBOUND Simulations are C structs underneath the Python, this function can pass the simulation by reference.
        Meaning, any changes made inside this function to the REBOUND simulation are permanent. This can be avoided by specifying overwrite=False.

        Parameters
        ----------
        sim : the REBOUND Simulation (star and planets) that will experience the flyby star.
        stars: an AIRBALL Stars object (containing multiple stars).
        times: an array of times for the stars to be added to the sim.

        rmax : the starting distance of the flyby star in units of AU
        overwrite: determines whether or not to return a copy of sim (overwrite=False) or integrate using the original sim (overwrite=True)
        plane: String/Int. The plane defining the orientation of the star, None, 'invariable', 'ecliptic', or Int. Default is None.
    '''
    message = "Integrating flybys concurrently may give unintuitive results. Use with caution."
    _warnings.warn(message, RuntimeWarning)

    # Do not overwrite given sim.
    overwrite = kwargs.get('overwrite', True)
    if not overwrite: sim = sim.copy()
    sim_units = _tools.rebound_units(sim)

    rmax = kwargs.get('rmax', 1e5*_u.au)
    plane = kwargs.get('plane')
    start_times = _tools.verify_unit(start_times, sim_units['time']).value
    hashes = kwargs.get('hashes', [f'flybystar{i}' for i in range(stars.N)])

    # Using the sim and the start times, compute the end times for the flyby stars.
    all_times = _numpy.zeros((stars.N, 2))
    for star_number, star in enumerate(stars):
        tmp_sim = sim.copy()
        hash = hashes[star_number]
        add_star_to_sim(tmp_sim, star, hash=hash, rmax=rmax, plane=plane)
        # Compute the time to periapsis for the flyby star from the current simulation time.
        tperi = tmp_sim.particles[hash].T - tmp_sim.t 
        end_time = start_times[star_number] + tmp_sim.t + 2*tperi
        all_times[star_number] = [start_times[star_number], end_time]

    # Sort the event times sequentially.
    all_times = all_times.flatten()
    event_order = _numpy.argsort(all_times)
    max_event_number = len(all_times)
    
    # Integrate the flybys, adding and removing them at the appropriate times.
    event_number = 0
    while event_number < max_event_number:
        event_index = event_order[event_number]
        star_number = event_index//2
        sim.integrate(all_times[event_index])
        if event_index%2 == 0: add_star_to_sim(sim, stars[star_number], hash=hashes[star_number], rmax=rmax, plane=plane)
        else: remove_star_from_sim(sim, hash=hashes[star_number])
        event_number += 1
    return sim



def _hybrid_successive_flybys(sim, stars, rmax=1e5, crossoverFactor=30, overwrite=False, heartbeat=None, particle_index=1, plane=None):
    '''
        Simulate a stellar flyby to a REBOUND simulation.

        Because REBOUND Simulations are C structs underneath the Python, this function can pass the simulation by reference.
        Meaning, any changes made inside this function to the REBOUND simulation are permanent. This can be avoided by specifying overwrite=False.
        This function assumes that you are using a WHFAST integrator with REBOUND.
        Uses IAS15 (instead of WHFast) for the closest approach if b < planet_a * crossoverFactor

        Parameters
        ----------
        sim : the REBOUND Simulation (star and planets) that will experience the flyby star
        star: a AIRBALL Star object

        rmax : the starting distance of the flyby star in units of AU
        crossoverFactor: the value for when to switch integrators if hybrid=True
        overwrite: determines whether or not to return a copy of sim (overwrite=False) or integrate using the original sim (overwrite=True)
        integrator: sets the integrator for before and after the hybrid switch (for example, if you want to use WHCKL instead of WHFast)
        heartbeat: sets a heartbeat function
        plane: String/Int. The plane defining the orientation of the star, None, 'invariable', 'ecliptic', or Int. Default is None.
    '''

    # Do not overwrite given sim.
    if not overwrite: sim = sim.copy()
    if heartbeat is not None: sim.heartbeat = heartbeat
    sim_units = _tools.rebound_units(sim)

    output = None
    with _tempfile.NamedTemporaryFile() as tmp:
        sim.simulationarchive_snapshot(tmp.name, deletefile=True)
        for star_number, star in enumerate(stars):
            index = int(particle_index)
            hash = f'flybystar{star_number}'
            star_vars = add_star_to_sim(sim, star, rmax=rmax, plane=plane, hash=hash)

            tperi = sim.particles[hash].T - sim.t # Compute the time to periapsis for the flyby star from the current time.

            # Integrate the flyby. Start at the current time and go to twice the time to periapsis.
            switch, tIAS15 = _time_to_periapsis_from_crossover_point(sim, sim_units, crossoverFactor, index, star_vars)
            if switch:
                t_switch = sim.t + tperi - tIAS15.value
                t_switch_back = sim.t + tperi + tIAS15.value
                t_end = sim.t + 2*tperi

                dt = sim.dt
                dt_frac = sim.dt/sim.particles[1].P

                _integrate_with_whckl(sim, t_switch, dt, dt_frac)
                sim.simulationarchive_snapshot(tmp.name, deletefile=False)
                _integrate_with_ias15(sim, t_switch_back)
                sim.simulationarchive_snapshot(tmp.name, deletefile=False)
                _integrate_with_whckl(sim, t_end, dt, dt_frac)

            else: _integrate_with_whckl(sim, tmax=(sim.t + 2*tperi), dt=sim.dt, dt_frac=sim.dt/sim.particles[1].P)

            # Remove the flyby star.
            remove_star_from_sim(sim, hash=hash)
            sim.simulationarchive_snapshot(tmp.name, deletefile=False)
        output = _rebound.SimulationArchive(tmp.name)
    return output

def _hybrid_successive_flybys_parallel(sims, stars, **kwargs):
    '''
        Run serial flybys in parallel.

        Parameters
        ---------------
        sims : A list of REBOUND Simulations.
            REBOUND simulations to integrate flybys with. If only one simulation is given, then AIRBALL will duplicate it to match the number of Stars given. Required.
        stars : A list of AIRBALL Stars.
            The objects that will flyby the given REBOUND simulations. Required.

        crossoverFactor : Float.
            The value for when to switch to IAS15 as a multiple of sim.particles[1].a Default is 30.
        overwrite : True/False.
            Sets whether or not to return new simulation objects or overwrite the given ones. Default is True, meaning the same simulation object will be returned. This keeps all original pointers attached to it.
        rmax : Float.
            The starting distance of the flyby object (in units of the REBOUND Simulation). Default is 1e5.
        plane : String/Int.
            The plane defining the orientation of the star, None, 'invariable', 'ecliptic', or Int. Default is None.
        particle_index : Int.
            The simulation particle index to define the crossoverFactor with respect to. Default is 1.

        inds : array_like
            An array of indices to determine which sims and stars to integrate. Default is all of them.
        n_jobs : Integer.
            The number of jobs per CPU to run in parallel. Default is -1.
        verbose : Integer.
            The amount of details to display for the parallel jobs. Default is 0.

        Returns
        -------------
        hybrid_flybys : list
            List of REBOUND simulations that experienced a flyby.
    '''

    _warnings.warn("This function has not been thoroughly tested. Use with caution.", RuntimeWarning)

    Nruns = 0
    try:
        Nruns = len(sims)
        if Nruns != len(stars): raise Exception('Sims and stars are unequal lengths')
    except Exception as err:
        # TypeError: object of type 'Simulation' has no len()
        raise err

    try:
        rmax = kwargs['rmax']
        if not _tools.isList(rmax): rmax = Nruns * [rmax]
        elif len(rmax) != Nruns: raise Exception('List arguments must be same length.')
    except KeyError: rmax = Nruns * [1e5]

    try:
        crossoverFactor = kwargs['crossoverFactor']
        if not _tools.isList(crossoverFactor): crossoverFactor = Nruns * [crossoverFactor]
        elif len(crossoverFactor) != Nruns: raise Exception('List arguments must be same length.')
    except KeyError: crossoverFactor = Nruns * [30]

    heartbeat = kwargs.get('heartbeat', None)
    inds = kwargs.get('inds', _numpy.arange(Nruns))
    overwrite = kwargs.get('overwrite', True)
    n_jobs = kwargs.get('n_jobs', -1)
    verbose = kwargs.get('verbose', 0)
    particle_index = kwargs.get('particle_index', 1)
    plane = kwargs.get('plane', None)

    sim_results = _joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
    _joblib.delayed(_hybrid_successive_flybys)(
        sim=sims[int(i)], stars=stars[int(i)], rmax=rmax[i], crossoverFactor=crossoverFactor[i], overwrite=overwrite,  particle_index=particle_index, plane=plane, heartbeat=heartbeat)
    for i in inds)

    return sim_results

def _hybrid_concurrent_flybys(sim, stars, times, rmax=1e5, crossoverFactor=30, overwrite=False, heartbeat=None, particle_index=1, plane=None, verbose=False):
    '''
        Simulate a stellar flyby to a REBOUND simulation.

        Because REBOUND Simulations are C structs underneath the Python, this function can pass the simulation by reference.
        Meaning, any changes made inside this function to the REBOUND simulation are permanent. This can be avoided by specifying overwrite=False.
        This function assumes that you are using a WHFAST integrator with REBOUND.
        Uses IAS15 (instead of WHFast) for the closest approach if b < planet_a * crossoverFactor

        Parameters
        ----------
        sim : the REBOUND Simulation (star and planets) that will experience the flyby star
        star: a AIRBALL Star object

        rmax : the starting distance of the flyby star in units of AU
        crossoverFactor: the value for when to switch integrators if hybrid=True
        overwrite: determines whether or not to return a copy of sim (overwrite=False) or integrate using the original sim (overwrite=True)
        integrator: sets the integrator for before and after the hybrid switch (for example, if you want to use WHCKL instead of WHFast)
        heartbeat: sets a heartbeat function
        plane: String/Int. The plane defining the orientation of the star, None, 'invariable', 'ecliptic', or Int. Default is None.
    '''

    _warnings.warn("This function has not been thoroughly tested. Use with caution.", RuntimeWarning)

    # Do not overwrite given sim.
    if not overwrite: sim = sim.copy()
    if heartbeat is not None: sim.heartbeat = heartbeat
    sim_units = _tools.rebound_units(sim)
    index = int(particle_index)

    times = _tools.verify_unit(times, sim_units['time']).value
    all_times = []
    for star_number, star in enumerate(stars):
        these_times = []
        tmp_sim = sim.copy()
        hash = f'tmp{star_number}'
        star_vars = add_star_to_sim(tmp_sim, star, rmax=rmax, plane=plane, hash=hash)

        tperi = times[star_number] + tmp_sim.particles[hash].T - tmp_sim.t # Compute the time to periapsis for the flyby star from the current time.
        these_times.append(times[star_number])
        # Integrate the flyby. Start at the current time and go to twice the time to periapsis.
        switch, tIAS15 = _time_to_periapsis_from_crossover_point(tmp_sim, sim_units, crossoverFactor, index, star_vars)
        if switch:
            these_times.append(times[star_number] + tmp_sim.t + tperi - tIAS15.value)
            these_times.append(times[star_number] + tmp_sim.t + tperi + tIAS15.value)
        else:
            these_times.append(_numpy.nan)
            these_times.append(_numpy.nan)

        these_times.append(times[star_number] + tmp_sim.t + 2*tperi)
        all_times.append(these_times)
    all_times = _numpy.array(all_times).flatten()
    max_event_number = len(all_times) - _numpy.sum(_numpy.isnan(all_times))
    event_order = _numpy.argsort(all_times)
    if verbose:
        tmpdic = {0 : f'ADD', 1 : f'start IAS15', 2 : f'end IAS15', 3 : f'REMOVE'}
        print([f'{tmpdic[i%4]} {i//4}' for i in event_order[:max_event_number]])

    useIAS15 = _numpy.array([False] * stars.N)
    def startUsingIAS15(i, IAS15_array): IAS15_array[i//4] = True
    def stopUsingIAS15(i, IAS15_array): IAS15_array[i//4] = False
    
    def function_map( i, v, sim, star, IAS15_array, plane, hash):
        if not _numpy.isnan(v): 
            map_i = i%4
            if map_i == 0: add_star_to_sim(sim, star, plane=plane, hash=hash)
            elif map_i == 1: startUsingIAS15(i, IAS15_array)
            elif map_i == 2: stopUsingIAS15(i, IAS15_array)
            elif map_i == 3: remove_star_from_sim(sim, hash=hash)
            else: pass
    
    output = None
    event_number = 0
    dt = sim.dt
    dt_frac = sim.dt/sim.particles[1].P
    with _tempfile.NamedTemporaryFile() as tmp:
        sim.simulationarchive_snapshot(tmp.name, deletefile=True)
        while event_number < max_event_number:
            event_index = event_order[event_number]
            star_number = event_index//4
            if _numpy.any(useIAS15): _integrate_with_ias15(sim, all_times[event_index])
            else: _integrate_with_whckl(sim, all_times[event_index], dt, dt_frac)
            function_map(event_index, all_times[event_index], sim, stars[star_number], useIAS15, plane, hash=f'flybystar{star_number}')
            sim.simulationarchive_snapshot(tmp.name, deletefile=False)
            event_number += 1
        output = _rebound.SimulationArchive(tmp.name)

    return output
