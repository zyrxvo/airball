import numpy as _np
import joblib as _joblib
import warnings as _warnings
from scipy.special import j0 as _j0,jv as _jv

from . import tools as _tools
from . import units as _u
from .core import add_star_to_sim, _rotate_into_plane

############################################################
################## Energy Estimates ########################
############################################################

def binary_energy(sim, particle_index=1):
    """
    Calculate the energy of a binary system, $-(G M m)/(2 a)$.

    Args:
        sim (Simulation): The simulation with two bodies, a central star and a planet.
        particle_index (int, optional): The index of the particle in the simulation to calculate the energy for. Default is 1.

    Returns:
        energy (Quantity): The energy of the binary system.

    Example:
        ```python
        import rebound
        import airball
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=5e-5, a=30., e=0.1, inc=0.1)
        energy = airball.analytic.binary_energy(sim)
        ```
    """
    index = int(particle_index)
    unit_set = _tools.rebound_units(sim)
    G = (sim.G * unit_set.length**3 / unit_set.mass / unit_set.time**2)
    p = sim.particles
    if p[index].m == 0:
        message = "Planet has zero mass therefore the binary energy is zero. This may cause divide by zero error when calculating the relative change in energy."
        _warnings.warn(message, RuntimeWarning)
    return (-G * p[0].m * unit_set.mass * p[index].m * unit_set.mass / (2. * p[index].a * unit_set['length'])).decompose(list(unit_set.values()))
    
def energy_change_adiabatic_estimate(sim, star, averaged=False, particle_index=1):
    """
    An analytical estimate for the change in energy of a binary system due to a stellar flyby.

    This function is based on the conclusions of [Roy & Haddow (2003)](https://ui.adsabs.harvard.edu/abs/2003CeMDA..87..411R/abstract) and [Heggie (2006)](https://ui.adsabs.harvard.edu/abs/2006fbp..book...20H/abstract). The orbital element angles of the flyby star are determined with respect to the plane defined by the binary orbit.

    Args:
        sim (Simulation): The simulation to calculate the energy change for.
        star (Star or Stars): The star or stars that are flying by the binary system.
        averaged (bool, optional): Whether to return the averaged energy change. Default is False.
        particle_index (int, optional): The index of the particle in the simulation to calculate the energy change for. Default is 1.

    Returns:
        result (Quantity): The estimated change in energy.

    Example:
        ```python
        import rebound
        import airball
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=5e-5, a=30., e=0.1, inc=0.1)
        star = airball.Star(m=1., b=500, v=5)
        energy_change = airball.energy_change_adiabatic_estimate(sim, star)
        ```
    """
    index = int(particle_index)
    units = _tools.rebound_units(sim)
    t0 = 0*units.time
    G = (sim.G * units.length**3 / units.mass / units.time**2)

    sim = sim.copy()
    _rotate_into_plane(sim, plane=index)
    # add_star_to_sim(sim, star, hash='flybystar', rmax=0) # Initialize Star at perihelion
    sim.move_to_hel()
    
    p = sim.particles
    m1, m2, m3 = p[0].m * units.mass, p[index].m * units.mass, star.mass # redefine the masses for convenience
    M12 = m1 + m2 # total mass of the binary system
    M123 = m1 + m2 + m3 # total mass of all the objects involved
    
    mu = G * (_tools.system_mass(sim)  * units.mass + m3)
    es = _tools.calculate_eccentricity(sim, star)
    
    a, e = p[index].a * units.length, p[index].e # redefine the orbital elements of the planet for convenience
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
    if averaged: noncircular_result = (prefactor * (e1 + e2 * (1 - e*e) + 2 * e4 * _np.sqrt(1 - e*e))).decompose(list(units.values()))
    else: noncircular_result = (prefactor * ( term1 + term2 + term3)).decompose(list(units.values()))

    # Case: Circular Binary

    # Compute the prefactor and terms of the calculation done by Roy & Haddow (2003)
    prefactor = (-_np.sqrt(_np.pi)/8.0) * ((G*m1*m2*m3)/M12) * ((a*a*a)/(q*q*q*q)) * f1 * k**(3.5) * _np.exp((-2.0*k/3.0)*f2) * (m2/M12 - m1/M12)
    with _u.set_enabled_equivalencies(_u.dimensionless_angles()):
        term1 = (1.0 + _np.cos(i)) * _np.sin(i)**2.0
        term2 = ( (_np.cos(w)**3.0) - 3.0 * (_np.sin(w)**2.0) * _np.cos(w) ) * _np.sin(n*t0)
        term3 = ( 3.0 * (_np.cos(w)**2.0) * _np.sin(w) - (_np.sin(w)**3.0)) * _np.cos(n*t0)
    
    circular_result = None
    if averaged: circular_result = (prefactor / _np.pi).decompose(list(units.values()))
    else: circular_result = (prefactor * term1 * (term2 + term3)).decompose(list(units.values()))

    return circular_result + noncircular_result

def energy_change_close_encounter_estimate(sim, star, particle_index=1):
    '''
    An analytical estimate for the change in energy of a binary system due to a flyby star. From Equation (4.7) of [Heggie (1975)](https://ui.adsabs.harvard.edu/abs/1975MNRAS.173..729H/abstract). The orbital element angles of the flyby star are determined with respect to the plane defined by the binary orbit.

    Args:
        sim (Simulation): The simulation with two bodies, a central star and a planet.
        star (Star): The star used to estimate the change in energy of a binary due to the flyby star
        particle_index (int, optional): The index of the particle in the simulation to calculate the energy for. Default is 1.

    Returns:
        result (Quantity): The change in energy of the binary system.

    Example:
        ```python
        import rebound
        from airball import Star
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=5e-5, a=30., e=0.1, inc=0.1)
        star = Star(m=1., b=100, v=5)
        energy_change = airball.analytic.energy_change_close_encounter_estimate(sim, star)
        ```
    '''
    index = int(particle_index)
    units = _tools.rebound_units(sim)
    
    def process_star(this_sim, this_star, this_index):
        s = this_sim.copy()
        add_star_to_sim(s, this_star, hash='flybystar', rmax=0, plane=this_index) # Initialize Star at perihelion
        return _tools.unit_vector(s.particles['flybystar'].xyz << units.length)
    
    if star.N == 1:
        x,y,z = process_star(sim, star, index)
    else:
        x,y,z = _np.asarray(_joblib.Parallel(n_jobs=-1)(_joblib.delayed(process_star)(sim, this_star, index) for this_star in star)).T
    c = _tools.cartesian_elements(sim, star, rmax=0)
    dat = _np.array([c['x'].value, c['y'].value, c['z'].value]).T << units.length
    G = (sim.G * units.length**3 / units.mass / units.time**2) # Newton's Gravitational constant

    m1, m2 = sim.particles[0].m * units.mass, sim.particles[index].m * units.mass # redefine the masses for convenience
    m3 = star.m
    M12 = m1 + m2 # total mass of the binary system
    M23 = m2 + m3 # total mass of the second and third bodies

    V = star.v # velocity of the star

    
    vx,vy,vz = sim.particles[index].vxyz << (units.length/units.time)

    with _u.set_enabled_equivalencies(_u.dimensionless_angles()):
        cosϕ = 1.0/_np.sqrt(1.0 + (((star.b**2.0)*(V**4.0))/((G*M23)**2.0)).decompose())
    
    prefactor = (-2.0 * m1 * m2 * m3)/(M12 * M23) * V * cosϕ
    t1 = -(x*vx + y*vy + z*vz)
    t2 = (m3 * V * cosϕ)/M23
    return (prefactor * (t1 + t2)).decompose(units.values())

def relative_energy_change_close_encounter_estimate(sim, star, particle_index=1):
    '''
     A convenience function for computing the analytical estimate for the relative change in energy of a binary system due to a close encounter with a flyby star. Combines [`energy_change_close_encounter_estimate`][airball.analytic.energy_change_close_encounter_estimate] and [`binary_energy`][airball.analytic.binary_energy] functions.

    Args:
        sim (Simulation): The simulation with two bodies, a central star and a planet.
        star (Star or Stars): The star or stars used to estimate the change in energy of a binary due to the flyby star
        particle_index (int, optional): The index of the particle in the simulation to calculate the energy for. Default is 1.

    Returns:
        energy_change (Quantity): The relative change in energy of the binary system.

    Example:
        ```python
        import rebound
        import airball
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=5e-5, a=30., e=0.1, inc=0.1)
        star = airball.Star(m=1., b=500, v=5)
        energy_change = airball.analytic.relative_energy_change_close_encounter_estimate(sim, star)
        ```

    '''
    return energy_change_close_encounter_estimate(sim, star, particle_index=particle_index)/binary_energy(sim, particle_index=particle_index)

def relative_energy_change(sim, star, averaged=False, particle_index=1):
    '''
     A convenience function for computing the analytical estimate for the relative change in energy of a binary system due to a flyby star. Combines [`energy_change_adiabatic_estimate`][airball.analytic.energy_change_adiabatic_estimate] and [`binary_energy`][airball.analytic.binary_energy] functions.

    Args:
        sim (Simulation): The simulation with two bodies, a central star and a planet.
        star (Star or Stars): The star or stars used to estimate the change in energy of a binary due to the flyby star
        averaged (bool, optional): Boolean indicator for averaging over the incident angles of the flyby star. Default is False.
        particle_index (int, optional): The index of the particle in the simulation to calculate the energy for. Default is 1.

    Returns:
        energy_change (Quantity): The relative change in energy of the binary system.

    Example:
        ```python
        import rebound
        import airball
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=5e-5, a=30., e=0.1, inc=0.1)
        star = airball.Star(m=1., b=500, v=5)
        energy_change = airball.analytic.relative_energy_change(sim, star)
        ```

    '''
    unit_set = _tools.rebound_units(sim)
    qs = star.q(sim)/(sim.particles[particle_index].a * unit_set.length)
    q_crit = 2**(2/3)
    if star.N == 1:
        if qs < q_crit: return energy_change_close_encounter_estimate(sim, star, particle_index=particle_index)/binary_energy(sim, particle_index=particle_index)
        else: return energy_change_adiabatic_estimate(sim, star, averaged=averaged, particle_index=particle_index)/binary_energy(sim, particle_index=particle_index)
    else:
        de_close = energy_change_close_encounter_estimate(sim, star, particle_index=particle_index)/binary_energy(sim, particle_index=particle_index)
        de = energy_change_adiabatic_estimate(sim, star, averaged=averaged, particle_index=particle_index)/binary_energy(sim, particle_index=particle_index)
        close_inds = qs < q_crit
        far_inds = ~close_inds
        results = _np.zeros(star.N)
        results[close_inds] = de_close[close_inds]
        results[far_inds] = de[far_inds]
        return results
    # return energy_change_adiabatic_estimate(sim=sim, star=star, averaged=averaged, particle_index=particle_index)/binary_energy(sim, particle_index=particle_index)

def parallel_relative_energy_change(sims, stars, averaged=False, particle_index=1):
    '''
        A convenience function for computing the analytical estimate for the relative change in energy of a binary system due to a flyby star. Combines [`energy_change_adiabatic_estimate`][airball.analytic.energy_change_adiabatic_estimate] and [`binary_energy`][airball.analytic.binary_energy] functions parallelized over the input parameters.

        Args:
            sims (list of Simulations): The simulation with two bodies, a central star and a planet.
            stars (Stars): The stars used to estimate the change in energy of a binary due to the flyby star
            averaged (bool, optional): Boolean indicator for averaging over the incident angles of the flyby star. Default is False.
            particle_index (int, optional): The index of the particle in the simulation to calculate the energy for. Default is 1.

        Returns:
            result (list of Quantities): The relative change in energy of the binary systems.

        Example:
            ```python
            import rebound
            import airball
            def setup():
                sim = rebound.Simulation()
                sim.add(m=1.)
                sim.add(m=5e-5, a=30., e=0.1, f='uniform')
                return sim
            stars = airball.Star(m=1., b=500, v=5, size=10)
            sims = [setup() for _ in range(stars.N)]
            energy_change = airball.analytic.parallel_relative_energy_change(sims, stars)
            ```
    '''
    unit_set = _tools.rebound_units(sims[0])
    qs = stars.q(sims[0])
    de = _np.array(_joblib.Parallel(n_jobs=-1)(_joblib.delayed(relative_energy_change)(sim=sims[i], star=stars[i], averaged=averaged, particle_index=particle_index) for i in range(stars.N)))
    de_close = _np.array(_joblib.Parallel(n_jobs=-1)(_joblib.delayed(relative_energy_change_close_encounter_estimate)(sim=sims[i], star=stars[i], particle_index=particle_index) for i in range(stars.N)))
    close_inds = qs/(sims[0].particles[particle_index].a * unit_set.length) < 2**(2/3)
    far_inds = ~close_inds
    results = _np.zeros(stars.N)
    results[close_inds] = de_close[close_inds]
    results[far_inds] = de[far_inds]
    return results


############################################################
################ Eccentricity Estimates ####################
############################################################


def eccentricity_change_close_encounter_estimate(sim, star, particle_index=1):
    '''
    An analytical estimate for the change in eccentricity of a binary system due to a flyby star. From Equation A1 of [Spurzem et al. (2009)](https://ui.adsabs.harvard.edu/abs/2009MNRAS.393..457S/abstract). The orbital element angles of the flyby star are determined with respect to the plane defined by the binary orbit.
    '''

    index = int(particle_index)
    unit_set = _tools.rebound_units(sim)

    p = sim.particles
    m1, m2, m3 = p[0].m * unit_set.mass, p[index].m * unit_set.mass, star.mass # redefine the masses for convenience
    M12 = m1 + m2 # total mass of the binary system
    M123 = m1 + m2 + m3 # total mass of all the objects involved

    q = star.q(sim) # compute the periapsis of the flyby star
    AB = _tools.hyperbolic_plane(sim, star)
    A,B = AB['A'], AB['B']
    ahat, bhat = [1,0,0], [0,1,0]

    # mu = G * (_tools.system_mass(sim)  * unit_set.mass + m3)
    # es = _tools.vinf_and_b_to_e(mu=mu, star_b=star.b, star_v=star.v)

    a, e = p[index].a * unit_set.length, p[index].e # redefine the orbital elements of the planet for convenience

    return ( ((-15.0*_np.pi*m3) / _np.sqrt(32.0 * M12*M123)) * ((a/q)**1.5) * (e * _np.sqrt(1.0 - e*e)) * ( (ahat @ A) * (bhat @ A) + (ahat @ B) * (bhat @ B) ) ).decompose(list(unit_set.values()))

def eccentricity_change_adiabatic_estimate(sim, star, averaged=False, particle_index=1, rmax=1e5*_u.au):
    '''
        An analytical estimate for the change in eccentricity of a binary system due to a flyby star. From Equations (7, 9, & 12) of Heggie & Rasio [(1996)](https://ui.adsabs.harvard.edu/abs/1996MNRAS.282.1064H/abstract). The orbital element angles of the flyby star are determined with respect to the plane defined by the binary orbit.

        Args:
            sim (Simulation): The simulation with two bodies, a central star and a planet.
            star (Star): The star used to estimate the change in energy of a binary due to the flyby star
            averaged (bool, optional): Boolean indicator for averaging over the incident angles of the flyby star. Default is False.
            particle_index (int, optional): The index of the particle in the simulation to calculate the energy for. Default is 1.
            rmax (float or Quantity, optional): Needed to determine the time when the star will be at perihelion. Default is $10^5$ AU.

        Returns:
            result (Quantity): The change in eccentricity of the binary system.

        Example:
            ```python
            import rebound
            import airball
            sim = rebound.Simulation()
            sim.add(m=1.)
            sim.add(m=5e-5, a=30., e=0.1, inc=0.1)
            star = airball.Star(m=1., b=500, v=5)
            eccentricity_change = airball.analytic.eccentricity_change_adiabatic_estimate(sim, star)
            ```
    '''

    index = int(particle_index)

    unit_set = _tools.rebound_units(sim)

    t0 = sim.t * unit_set.time
    G = (sim.G * unit_set.length**3 / unit_set.mass / unit_set.time**2)
    
    p = sim.particles
    m1, m2, m3 = p[0].m * unit_set.mass, p[index].m * unit_set.mass, star.mass # redefine the masses for convenience
    M12 = m1 + m2 # total mass of the binary system
    M123 = m1 + m2 + m3 # total mass of all the objects involved
    
    mu = G * (_tools.system_mass(sim)  * unit_set.mass + m3)
    es = _tools.vinf_and_b_to_e(mu=mu, star_b=star.b, star_v=star.v)
    
    a, e = p[index].a * unit_set.length, p[index].e # redefine the orbital elements of the planet for convenience
    # n = _np.sqrt(G*M12/a**3) # compute the mean motion of the planet
    
    w, W, i = star.omega, star.Omega, star.inc # redefine the orientation elements of the flyby star for convenience

    star_params = _tools.hyperbolic_elements(sim, star, rmax)
    tperi = star_params['T']
    Mperi = p[index].M + (p[index].n/unit_set.time) * (tperi - t0) # get the Mean anomaly when the flyby star is at perihelion
    f = _tools.M_to_f(p[index].e, Mperi.value) << _u.rad # get the true anomaly when the flyby star is at perihelion
    Wp = W + f
    V = star.v
    q = (- mu + _np.sqrt( mu**2. + star.b**2. * V**4.))/V**2. # compute the periapsis of the flyby star

    # Case: Non-Circular Binary

    prefactor = (-15.0/4.0) * m3 / _np.sqrt(M12*M123) * ((a/q)**1.5) * ((e * _np.sqrt(1.0 - e*e))/((1.0 + es)**1.5))
    with _u.set_enabled_equivalencies(_u.dimensionless_angles()):
        t1 = (_np.sin(i) * _np.sin(i) * _np.sin(2.0*W) * ( _np.arccos(-1.0/es) + _np.sqrt(es*es - 1.0) )).value
        t2 = ((1.0/3.0) * (1.0 + _np.cos(i)*_np.cos(i)) * _np.cos(2.0*w) * _np.sin(2.0*W)).value
        t3 = (2.0 * _np.cos(i) * _np.sin(2.0*w) * _np.cos(2.0*W) * ((es*es - 1.0)**1.5)/(es*es)).value

    if averaged:
        noncircular_result = (prefactor * ( ( _np.arccos(-1.0/es).value + _np.sqrt(es*es - 1.0) ) + (2.0/3.0) + (2.0 * ((es*es - 1.0)**1.5)/(es*es)) )).decompose(list(unit_set.values()))
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

    # Add the ABS together, deliberately choose a norm.
    return (_np.nan_to_num(circular_result) + _np.nan_to_num(noncircular_result) + _np.nan_to_num(exponential_result)).decompose()

def parallel_eccentricity_change_adiabatic_estimate(sims, stars, averaged=False, particle_index=1, rmax=1e5*_u.au):
    """
    An analytical estimate for the changes in eccentricity of binary systems due to a flyby stars.

    Args:
        sims (list of Simulations): the simulation with two bodies, a central star and a planet.
        stars (Stars): the star or stars used to estimate the change in energy of a binary due to the flyby star
        averaged (bool, optional): boolean indicator for averaging over the incident angles of the flyby star. Default is False.
        particle_index (int, optional): the index of the particle in the simulation to calculate the energy for. Default is 1.
        rmax (float or Quantity, optional): needed to determine the time when the star will be at perihelion. Default is $10^5$ AU.

    Returns:
        results (list of Quantities): The changes in eccentricity of the binary systems.

    Example:
        ```python
        import rebound
        import airball
        def setup():
            sim = rebound.Simulation()
            sim.add(m=1.)
            sim.add(m=5e-5, a=30., e=0.1, f='uniform')
            return sim
        stars = airball.Star(m=1., b=500, v=5, size=10)
        sims = [setup() for _ in range(stars.N)]
        eccentricity_changes = airball.analytic.parallel_eccentricity_change_adiabatic_estimate(sims, stars)
        ```
    """
    return _joblib.Parallel(n_jobs=-1)(_joblib.delayed(eccentricity_change_adiabatic_estimate)(sim=sims[i], star=stars[i], averaged=averaged, particle_index=particle_index, rmax=rmax) for i in range(stars.N))

def eccentricity_change_impulsive_estimate(sim, star, particle_index=1):
    '''
    An analytical estimate for the change in eccentricity of an eccentric binary system due to a flyby star. From Equation (28) of [Heggie & Rasio (1996)](https://ui.adsabs.harvard.edu/abs/1996MNRAS.282.1064H/abstract). The orbital element angles of the flyby star are determined with respect to the plane defined by the binary orbit.
    
    The base assumptions are that the flyby is coplanar, $q_\star \gg a$, and $v_\star \gg v_\mathrm{planet}$. It may work outside of this regime, but it is not guaranteed.
    
    TODO:
        Make sure that the time of perihelion is calculated correctly and that the phase of the planet is correct.

    Args:
        sim (Simulation): The simulation with two bodies, a central star and a planet.
        star (Star): The star used to estimate the change in energy of a binary due to the flyby star
        particle_index (int, optional): The index of the particle in the simulation to calculate the energy for. Default is 1.

    Returns:
        result (Quantity): The change in eccentricity of the binary system.

    Example:
        ```python
        import rebound
        import airball
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=5e-5, a=30., e=0.1, inc=0.1)
        star = airball.Star(m=1., b=500, v=5)
        eccentricity_change = airball.analytic.eccentricity_change_impulsive_estimate(sim, star)
        ```
    '''

    index = int(particle_index)
    units = _tools.rebound_units(sim)
    G = (sim.G * units.length**3 / units.mass / units.time**2)

    p = sim.particles
    m1, m2, m3 = p[0].m * units.mass, p[index].m * units.mass, star.mass # redefine the masses for convenience
    M12 = m1 + m2 # total mass of the binary system
    
    a = p[index].a * units.length # redefine the orbital elements of the planet for convenience
    V = star.v
    q = star.q(sim) # compute the periapsis of the flyby star
    c = _tools.cartesian_elements(sim, star, rmax=0) # Get the heliocentric coordinates of the flyby star at periapsis
    dat = _np.array([c['x'].value, c['y'].value, c['z'].value]).T << units.length
    theta = _tools.angle_between(p[index].xyz, dat)

    # print(f"{G=}, {M12=}, {m3=}, {V=}, {a=}, {q=}, {theta=}")
    return ((2.0 * _np.sqrt(G/M12) * (m3/V) * _np.sqrt(a*a*a) / (q*q)) * _np.abs(_np.cos(theta)) * _np.sqrt((_np.cos(theta)**2.0) + (4.0 * _np.sin(theta)**2.0))).decompose()

def parallel_eccentricity_change_impulsive_estimate(sims, stars, particle_index=1):
    """
    An analytical estimate for the changes in eccentricity of binary systems due to a flyby stars.

    Args:
        sims (list of Simulations): the simulation with two bodies, a central star and a planet.
        stars (Stars): the star or stars used to estimate the change in energy of a binary due to the flyby star
        particle_index (int, optional): the index of the particle in the simulation to calculate the energy for. Default is 1.

    Returns:
        results (list of Quantities): The changes in eccentricity of the binary systems.

    Example:
        ```python
        import rebound
        import airball
        def setup():
            sim = rebound.Simulation()
            sim.add(m=1.)
            sim.add(m=5e-5, a=30., e=0.1, f='uniform')
            return sim
        stars = airball.Star(m=1., b=500, v=5, size=10)
        sims = [setup() for _ in range(stars.N)]
        eccentricity_changes = airball.analytic.parallel_eccentricity_change_impulsive_estimate(sims, stars)
        ```
    """
    return _joblib.Parallel(n_jobs=-1)(_joblib.delayed(eccentricity_change_impulsive_estimate)(sim=sims[i], star=stars[i], particle_index=particle_index) for i in range(stars.N))

############################################################
################ Inclination Estimate ######################
############################################################

def inclination_change_adiabatic_estimate(sim, star, averaged=False, particle_index=1):
    '''
    An analytical estimate for the sine of the change in inclination of an eccentric binary system due to a flyby star, $\\sin(\\Delta i)$. This function is based on Equation (A5) of [Malmberg, Davies, & Heggie (2011)](https://ui.adsabs.harvard.edu/abs/2011MNRAS.411..859M/abstract), with the addition of a factor of $(1 + e_\\star)^{-1/2}$ to account for hyperbolic flybys. The orbital element angles of the flyby star are determined with respect to the plane defined by the binary orbit.

    Args:
        sim (Simulation): the simulation with two bodies, a central star and a planet.
        star (Star or Stars): The star or stars that are flying by the binary system.
        averaged (bool, optional): Whether to return the inclination change averaged over the incident angles. Default is False.
        particle_index (int, optional): the index of the particle in the simulation to calculate the inclination change for. Default is 1.

    Returns:
        result (Quantity): Sine of the estimated change in the inclination, $\\sin(\\Delta i)$.

    Example:
        ```python
        import rebound
        import airball
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=5e-5, a=30., e=0.1, inc=0.1)
        star = airball.Star(m=1., b=500, v=5)
        inclination_change = airball.analytic.inclination_change_adiabatic_estimate(sim, star)
        ```
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
    es = _tools.calculate_eccentricity(sim, star)
    
    a, e = p[index].a * unit_set['length'], p[index].e # redefine the orbital elements of the planet for convenience
    
    w, W, i = star.omega, star.Omega, star.inc # redefine the orientation elements of the flyby star for convenience

    V = star.v
    q = (- mu + _np.sqrt( mu**2. + star.b**2. * V**4.))/V**2. # compute the periapsis of the flyby star

    # Case: Non-Circular Binary

    prefactor = (3.0*_np.pi * m3 /8.0) * _np.sqrt(2.0/(M123 * M12 * (1.0 - e*e) * (1.0 + es))) * (a/q)**1.5
    with _u.set_enabled_equivalencies(_u.dimensionless_angles()):
        angles = (_np.sin(i) * _np.cos(i) * _np.sqrt( (1.0 + 3.0*e*e)**2.0 * _np.sin(W)**2.0 + (1.0 + e*e)**2.0 * _np.cos(W)**2.0 )).value

    if averaged:
        noncircular_result = (prefactor).decompose(list(unit_set.values()))
    else: noncircular_result = (prefactor * angles).decompose(list(unit_set.values()))

    return noncircular_result