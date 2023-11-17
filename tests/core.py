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
    star = airball.Star(m=0, b=1, v=1, inc=0, omega=0, Omega=0)
    sim = rebound.Simulation()
    sim.add(m=1)
    airball.add_star_to_sim(sim, star, rmax=1e5, hash='star')
    assert sim.N == 2

def test_add_star_to_empty_sim():
    star = airball.Star(m=0, b=1, v=1, inc=0, omega=0, Omega=0)
    sim = rebound.Simulation()
    with pytest.raises(AttributeError):
        airball.add_star_to_sim(sim, star, rmax=1e5, hash='star')

def test_remove_star():
    sim = rebound.Simulation()
    with pytest.raises(RuntimeError):
        airball.core.remove_star_from_sim(sim, 'star')


################################################
################################################
#############  FLYBY TESTS  ####################
################################################
################################################

def test_basic_flyby():
    star = airball.Star(m=0, b=1, v=1, inc=0, omega=0, Omega=0)
    sim = rebound.Simulation()
    sim.add(m=1)
    simout = airball.flyby(sim, star)
    assert sim.N == simout.N
