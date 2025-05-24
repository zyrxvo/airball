import pytest
import rebound
import airball
import airball.units as u
import numpy as np

################################################
################################################
##########  INITIALIZATION TESTS  ##############
################################################
################################################

def test_add_star():
    star = airball.Star(m=1, b=1, v=1, inc=1, omega=1, Omega=1)
    sim = rebound.Simulation()
    sim.add(m=1)
    RMAX = 1e5*u.au
    el1 = airball.tools.hyperbolic_elements(sim, star, rmax=RMAX, values_only=True)
    airball.add_star_to_sim(sim, star, rmax=RMAX, hash='star')
    el0 = sim.orbits(primary=sim.particles[0])[0]
    assert sim.N == 2
    assert np.isclose(el0.a, el1['a'])
    assert np.isclose(el0.e, el1['e'])
    assert np.isclose(el0.inc, el1['inc'])
    assert np.isclose(el0.omega, el1['omega'])
    assert np.isclose(el0.Omega, el1['Omega'])

def test_add_star_to_empty_sim():
    star = airball.Star(m=1, b=1, v=1, inc=0, omega=0, Omega=0)
    sim = rebound.Simulation()
    with pytest.raises(ValueError, match="Primary has no mass."):
        airball.add_star_to_sim(sim, star, rmax=1e5, hash='star')

def test_remove_star():
    sim = rebound.Simulation()
    sim.add(m=0, hash='flybystar')
    airball.core.remove_star_from_sim(sim, 'flybystar')
    assert sim.N == 0

    sim = rebound.Simulation()
    with pytest.raises(RuntimeError):
        airball.core.remove_star_from_sim(sim, 'star')


################################################
################################################
#############  FLYBY TESTS  ####################
################################################
################################################

def test_basic_flyby():
    star = airball.Star(m=1, b=1, v=1, inc=0, omega=0, Omega=0)
    sim = rebound.Simulation()
    sim.add(m=1)
    simout = airball.flyby(sim, star)
    assert sim == simout
    simout = airball.flyby(sim, star, overwrite=False)
    assert sim != simout
