import rebound
import numpy as np
import matplotlib.pyplot as plt

twopi = 2.*np.pi
kms_to_auyr2pi = 0.03357365989646266 # 1 km/s to AU/Yr2Pi
pc3_to_au3 = ((np.pi**3)/272097792000000000) # 1 parsec^-3 to AU^-3

def vinf_and_b_to_e(Mc, m, Ms, b, vinf):
    '''
        Using the impact parameter to convert from the relative velocity at infinity between the two stars to the eccentricity of the flyby star.
        Equation (2) from Spurzem et al. (2009) https://ui.adsabs.harvard.edu/abs/2009ApJ...697..458S/abstract

        Parameters
        ----------
        Mc : mass of the central star (ex. the Sun)
        m : mass of the planet orbiting the central star
        Ms : mass of the flyby star
        b :  impact parameter of the flyby star
        vinf : the relative velocity at infinity between the central star and the flyby star (hyperbolic excess velocity)
    '''
    G = 1 # Use units where G = 1, Msun, AU, Yr2Pi
    v = vinf*kms_to_auyr2pi # Convert velocity units from km/s to AU/Yr2Pi
    M123 = Mc + m + Ms # Compute sum of the masses
    numerator = b * v**2.
    denominator = G * M123
    return np.sqrt(1 + (numerator/denominator)**2.)

def flyby(sim, star_mass=1, star_b=100,  star_e=None, star_v=None, star_omega='uniform', star_Omega='uniform', star_inc='uniform', showflybysetup=False):
    '''
        Simulate a stellar flyby to a REBOUND simulation.
        
        Because REBOUND Simulations are C structs underneath the Python, this function passes the simulation by reference. 
        Any changes made inside this function to the REBOUND simulation are permanent.
        This function assumes that you are using a WHFAST integrator with REBOUND.
    '''
    if star_e is None and star_v is not None:
        # If `star_v` is defined convert it to eccentricity.
        # Assumes that `star_v` is in units of km/s.
        sun_mass = sim.particles[0].m
        planet_mass = sim.particles[1].m
        e = vinf_and_b_to_e(Mc=sun_mass, m=planet_mass, Ms=star_mass, b=star_b, vinf=star_v)
    elif star_e is not None and star_v is None:
        # Simply use the eccentricity if it is defined.
        e = star_e
    elif star_e is not None and star_v is not None: raise AssertionError('Cannot specify an eccentricity and a velocity for the perturbing star.')
    else: raise AssertionError('Specify either an eccentricity or a velocity for the perturbing star.')
    
    #################################################
    ## Cacluation of Flyby Star Initial Conditions ## 
    #################################################
    
    # Calculate the orbital elements of the flyby star.
    rmax = 1e6 # This is the starting distance of the flyby star in AU
    a = -star_b/np.sqrt(e**2. - 1.) # Compute the semi-major axis of the flyby star
    l = -a*(e*e-1.) # Compute the semi-latus rectum of the hyperbolic orbit (-a because the semi-major axis is negative)
    f = np.arccos((l/rmax-1.)/e) # Compute the true anomaly
    
    # Calculate half of the integration time for the flyby star.
    E = np.arccosh((np.cos(f)+e)/(1.+e*np.cos(f))) # Compute the eccentric anomaly
    M = e * np.sinh(E)-E # Compute the mean anomaly
    mu = sim.G * (np.sum([p.m for p in sim.particles]) + star_mass)
    tperi = M/np.sqrt(mu/(-a*a*a)) # Compute the time to pericentre (-a because the semi-major axis is negative)

    #################################################
    
    # Add the flyby star to the simulation. 
    sim.move_to_hel() # Move the system into the heliocentric frame of reference.
    sim.add(m=star_mass, a=a, e=e, f=-f, omega=star_omega, Omega=star_Omega, inc=star_inc, hash='flybystar')
    sim.ri_whfast.recalculate_coordinates_this_timestep = 1 # Because a new particle was added, we need to tell REBOUND to recalculate the coordinates.
    sim.move_to_com() # Move the system back into the centre of mass/momuntum frame for integrating.
    
    # Integrate the flyby. Start at the current time and go to twice the time to pericentre.
    if showflybysetup:
        t1 = sim.t + tperi
        t2 = sim.t + 2*tperi
        sim.integrate(t1)
        print('During the Flyby')
        sim.move_to_hel()
        rebound.OrbitPlot(sim, color=True, slices=True);
        plt.show()
        sim.move_to_com()
        sim.integrate(t2)
    else:
        sim.integrate(sim.t + 2.*tperi)
    
    # Remove the flyby star. 
    sim.remove(hash='flybystar')
    sim.ri_whfast.recalculate_coordinates_this_timestep = 1 # Because a particle was removed, we need to tell REBOUND to recalculate the coordinates.
    sim.move_to_com() # Readjust the system back into the centre of mass/momuntum frame for integrating.