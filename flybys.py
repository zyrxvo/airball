import rebound
import numpy as np
import matplotlib.pyplot as plt

################################
###### Useful Constants ########
################################

twopi = 2.*np.pi
kms_to_auyr2pi = 0.03357365989646266 # 1 km/s to AU/Yr2Pi
pc3_to_au3 = ((np.pi**3)/272097792000000000) # 1 parsec^-3 to AU^-3




############################################################
### Stellar Mass generators using Initial Mass Functions ###
############################################################

def imf_gen_1(size):
    '''
        Generate stellar mass samples for single star systems between 0.01 and 1.0 Solar Mass.
        
        Computed using the inverted cummulative probability distribution (CDF) from the initial mass function (IMF) given in equation (17) by Chabrier (2003) https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C/abstract

        Parameters
        ----------
        size: the number of samples to draw.
    '''
    n = int(size)
    u = uniform.rvs(size=n)
    return 0.08*np.exp(2.246879476250902 * erfinv(-0.8094067254228074 + 1.6975098420629455*u))

def imf_gen_10(size):
    '''
        Generate stellar mass samples for single star systems between 0.01 and 10.0 Solar Masses.
        
        Computed using the inverted cummulative probability distribution (CDF) by smoothly combining the initial mass function (IMF) given in equation (17) by Chabrier (2003) https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C/abstract for stars less than 1.0 Solar Mass with the standard power-law IMF from Salpeter (1955) https://ui.adsabs.harvard.edu/abs/1955ApJ...121..161S/abstract for stars more than 1.0 Solar Mass.

        Parameters
        ----------
        size: the number of samples to draw.
    '''
    n = int(size)
    u = uniform.rvs(size=n)
    return np.where(u > 0.9424222533172513, 0.11575164791201686 / (1.0030379829867349 - u)**(10/13.), 0.08*np.exp(2.246879476250902 * erfinv(-0.8094067254228074 + 1.801220032833315*u)))

def imf_gen_100(size):
    '''
        Generate stellar mass samples for single star systems between 0.01 and 100.0 Solar Masses.
        
        Computed using the inverted cummulative probability distribution (CDF) by smoothly combining the initial mass function (IMF) given in equation (17) by Chabrier (2003) https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C/abstract for stars less than 1.0 Solar Mass with the standard power-law IMF from Salpeter (1955) https://ui.adsabs.harvard.edu/abs/1955ApJ...121..161S/abstract for stars more than 1.0 Solar Mass.

        Parameters
        ----------
        size: the number of samples to draw.
    '''
    n = int(size)
    u = uniform.rvs(size=n)
    return np.where(u > 0.9397105089399359, 0.11549535807627886 / (1.0001518217134586 - u)**(10/13.), 0.08*np.exp(2.246879476250902 * erfinv(-0.8094067254228074 + 1.8064178551944312*u)))





############################################################
################### Helper Functions #######################
############################################################

def vinf_and_b_to_e(mu, b, vinf):
    '''
        Using the impact parameter to convert from the relative velocity at infinity between the two stars to the eccentricity of the flyby star.
        Equation (2) from Spurzem et al. (2009) https://ui.adsabs.harvard.edu/abs/2009ApJ...697..458S/abstract

        Parameters
        ----------
        mu : the total mass of the system (Sun, planets, and flyby star) times the gravitational constant G in units where G = 1, Msun, AU, Yr2Pi
        b :  impact parameter of the flyby star in units of AU
        vinf : the relative velocity at infinity between the central star and the flyby star (hyperbolic excess velocity) in units of km/s
    '''
    v = vinf*kms_to_auyr2pi # Convert velocity units from km/s to AU/Yr2Pi
    numerator = b * v**2.
    return np.sqrt(1 + (numerator/mu)**2.)




############################################################
#################### Flyby Functions #######################
############################################################


def flyby(sim, star_mass=1, star_b=100,  star_e=None, star_v=None, star_omega='uniform', star_Omega='uniform', star_inc='uniform', showflybysetup=False):
    '''
        Simulate a stellar flyby to a REBOUND simulation.
        
        Because REBOUND Simulations are C structs underneath the Python, this function passes the simulation by reference. 
        Any changes made inside this function to the REBOUND simulation are permanent.
        This function assumes that you are using a WHFAST integrator with REBOUND.
    '''
    
    mu = sim.G * (np.sum([p.m for p in sim.particles]) + star_mass)
    if star_e is None and star_v is not None:
        # If `star_v` is defined convert it to eccentricity.
        # Assumes that `star_v` is in units of km/s.
        sun_mass = sim.particles[0].m
        planet_mass = sim.particles[1].m
        e = vinf_and_b_to_e(mu=mu, b=star_b, vinf=star_v)
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