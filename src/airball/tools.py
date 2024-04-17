import numpy as _np
import rebound as _rebound
import joblib as _joblib
import warnings as _warnings
from .units import UnitSet as _UnitSet
from . import units as _u
from mpmath import mp as _mp

# Set the precision
_mp.dps = 50  # 50 digits of precision

twopi = 2.*_np.pi

############################################################
################### Helper Functions #######################
############################################################

def rotate_into_plane(sim, plane):
    '''
    Rotates the simulation into the specified plane.

    Args:
        sim (Simulation): The REBOUND Simulation containing the star and planets that will experience a flyby.
        plane (str, int): The plane defining the orientation of the star: None, 'invariable', 'ecliptic', or int.

    Returns:
        rotation (Rotation): The rotation that was applied to the simulation.
    '''
    int_types = (int, _np.integer)
    rotation = _rebound.Rotation.to_new_axes(newz=[0,0,1])
    if plane is not None:
        # Move the system into the chosen plane of reference. TODO: Make sure the angular momentum calculations don't include other flyby stars.
        if plane == 'invariable': rotation = _rebound.Rotation.to_new_axes(newz=sim.angular_momentum())
        elif plane == 'ecliptic': rotation = _rebound.Rotation.to_new_axes(newz=calculate_angular_momentum(sim)[3]) # Assumes Earth is particle 3. 0-Sun, 1-Mecury, 2-Venus, 3-Earth, ...
        elif isinstance(plane, int_types):
            p = sim.particles[int(plane)]
            rotation = (_rebound.Rotation.orbit(Omega=p.Omega, inc=p.inc, omega=p.omega)).inverse()
    sim.rotate(rotation)
    return rotation

# Implemented from StackOverflow: https://stackoverflow.com/a/14314054
def moving_average(a, n=3, method=None) :
    '''
    Compute the moving average of an array of numbers using the nearest n elements.
    Adapted from [StackOverflow](https://stackoverflow.com/a/14314054).
    
    The options for handling NaN values are: `'nn'` (nearest neighbor), `'nan'` (ignore NaNs), and `None`. The default is `None` which uses `numpy.cumsum`. The `'nn'` method is slower than `'nan'` but attempts to replace the NaN values with the average of the adjacent values. Thus, if the adjacent values are NaN, then it will also return NaN.

    Args:
      a (array): The array of numbers to compute the moving average of.
      n (int): The number of elements to use in the moving average.
      method (str): The method to use for handling NaN values. 

    Returns:
      result (ndarray): The moving average of the array of numbers.
    
    Example:
      ```python
      import airball
      import numpy as np
      a = np.array([1,2,3,4,5,6,7,8,9,10])
      print(airball.tools.moving_average(a, n=3)) # [2. 3. 4. 5. 6. 7. 8. 9.]
      ```
    
  '''
    if method == 'nan': ret = _np.nancumsum(a)
    elif method == 'nn':
        bool = _np.isnan(a)
        inds = _np.arange(len(a))[bool]
        ret = a.copy()
        for i in inds:
            ret[i] = (ret[i-1 if i-1 > 0 else i+1] + ret[i+1 if i+1 < len(a) else i-1])/2.0
        ret = _np.cumsum(ret)
    else: ret = _np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# Implemented from StackOverflow: https://stackoverflow.com/a/33585850
def moving_median(arr, n=3, method=None):
    '''Compute the moving median of an array of numbers using the nearest n elements. Adapted from [StackOverflow](https://stackoverflow.com/a/33585850).

    The options for handling NaN values are: `'nn'` (nearest neighbor), `'nan'` (ignore NaNs), and `None`. The default is `None` which uses `numpy.cumsum`. The `'nn'` method is not implemented and defaults to `'nan'`.

    Args:
      a (array): The array of numbers to compute the moving median of.
      n (int): The number of elements to use in the moving median.
      method (str): The method to use for handling NaN values. 

    Returns:
      result (ndarray): The moving median of the array of numbers.
    
    Example:
      ```python
      import airball
      import numpy as np
      a = np.array([1,2,3,4,5,6,7,8,9,10])
      print(airball.tools.moving_median(a, n=3)) # [2. 3. 4. 5. 6. 7. 8. 9.]
      ```
    
    '''
    idx = _np.arange(n) + _np.arange(len(arr)-n+1)[:,None]
    if method == 'nan' or method == 'nn': return _np.nanmedian(arr[idx], axis=1)
    else: return _np.median(arr[idx], axis=1)

def save_as_simulationarchive(filename, sims, delete_file=True):
    '''
    Saves a list of REBOUND Simulations as a SimulationArchive.
    '''
    for i,s in enumerate(sims):
        s.save_to_file(filename, delete_file=(delete_file if i == 0 else False))

def notNone(a):
    """Returns True if array a contains at least one element that is not None. Returns False otherwise., Implemented from REBOUND particle.py"""
    return a.count(None) != len(a)

def hasTrue(a):
    """Returns True if array a contains at least one element that is True. Returns False otherwise."""
    return a.count(True) > 0

def numberOfElementsReturnedBySlice(start, stop, step):
   '''Returns the number of elements returned by the slice(start, stop, step) function.'''
   return (stop - start) // step

def _integrate(sim, tmax):
    sim.integrate(tmax)
    return sim

def integrate(sims, tmaxes, n_jobs=-1, verbose=0):
    '''
    Integrates the provided list of REBOUND Simulations to the provided times in a parallelized manner. The parallalization uses the joblib package, so the returned list of Simulations will be copies of the original Simulations. The original Simulations will **not** be modified.

    Args:
      sims (list): A list of REBOUND Simulations.
      tmaxes (list): A list of times to integrate each Simulation to.
      n_jobs (int): The number of cores to use for parallelization. Default is -1 which uses all available cores.
      verbose (int): The verbosity level. Default is 0 which is silent.
    
    Returns:
      sim_results (list): A list of the integrated REBOUND Simulations.
    '''
    sim_results = _joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
        _joblib.delayed(_integrate)(sim=sims[int(i)], tmax=tmaxes[int(i)]) for i in range(len(sims)))
    return sim_results

def hist(arr, bins=10, normalize=False, density=False, wfac=1):
    '''
        Performs a histogram of the provided array over a linearly spaced range of the data using the provided number of bins. The histogram is normalized by the area under the curve if `normalize=True`. The width of the bins can be altered by the provided factor `wfac`. Implemented from [StackOverflow](https://stackoverflow.com/a/30555229).

        Args:
          arr (array): The array to histogram.
          bins (int): The number of bins to use.
          normalize (bool): Whether to normalize the histogram.
          density (bool): Whether to return the density of the histogram.
          wfac (float): A factor to alter the width of the bins.
        
        Returns:
          x (array): The bin centers.
          y (array): The histogram values.
          w (float): The width of the bins.
    '''
    # """Return pairwise geometric means of adjacent elements."""
    geometric_means = lambda a: _np.sqrt(a[1:] * a[:-1])
    
    astart = _np.min(arr)
    aend = _np.max(arr)
    arange = _np.linspace(astart, aend, bins+1, endpoint=True)
    
    y,b = _np.histogram(arr, bins=arange, density=density)
    x = geometric_means(b)
    w = wfac * _np.mean(x[1:] - x[:-1])
    
    if normalize: return x, y/_np.trapz(y,x), w
    else: return x,y,w

def hist10(arr, bins=10, normalize=False, density=False, wfac=1):
    '''
        Performs a histogram of the provided array over a logarithmically spaced range of the data using the provided number of bins. The histogram is normalized by the area under the curve if `normalize=True`. The width of the bins can be altered by the provided factor `wfac`. Implemented from [StackOverflow](https://stackoverflow.com/a/30555229).

        Args:
          arr (array): The array to histogram.
          bins (int): The number of bins to use.
          normalize (bool): Whether to normalize the histogram.
          density (bool): Whether to return the density of the histogram.
          wfac (float): A factor to alter the width of the bins.
        
        Returns:
          x (array): The bin centers.
          y (array): The histogram values.
          w (float): The width of the bins.
    '''
    # """Return pairwise geometric means of adjacent elements."""
    geometric_means = lambda a: _np.sqrt(a[1:] * a[:-1])
    
    astart = _np.log10(_np.min(arr)/2)
    aend = _np.log10(_np.max(arr)*2)
    arange = _np.logspace(astart, aend, bins+1, endpoint=True)
    
    y,b = _np.histogram(arr, bins=arange, density=density)
    x = geometric_means(b)
    w = wfac * x*_np.mean((x[1:] - x[:-1])/x[:-1])
    
    if normalize: return x, y/_np.trapz(y,x), w
    else: return x,y,w

def unit_vector(vector):
    """ 
    Returns the unit vector of the vector.
    Fails if the vector is a list of Quantity objects.
    
    Args:
      vector (array): The vector to convert to a unit vector.

    Returns:
      vector (array): The unit vector of the vector.
    """
    try:
        if len(vector.shape) > 1: return vector / _np.linalg.norm(vector, axis=1)[:,None]
        else: return vector / _np.linalg.norm(vector)
    except AttributeError: return vector / _np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'. Implemented from [StackOverflow](https://stackoverflow.com/a/13849249/71522).

    Args:
      v1 (array): The first vector.
      v2 (array): The second vector.
    
    Returns:
      theta (float): The angle between the two vectors in units of radians.
    
    Example:
      ```python
      angle_between((1, 0, 0), (0, 1, 0)) # 1.5707963267948966
      angle_between((1, 0, 0), (1, 0, 0)) # 0.0
      angle_between((1, 0, 0), (-1, 0, 0)) # 3.141592653589793
      ```
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return _np.arccos(_np.dot(v1_u, v2_u.T))

def mod2pi(f):
    '''Converts an angle to the range [0, 2pi). Implemented from REBOUND using Numpy to handle vectorization.'''
    return _np.mod(twopi + _np.mod(f, twopi), twopi)

def M_to_E(e, M):
  '''Converts mean anomaly to eccentric anomaly. Implemented from REBOUND using Numpy to handle vectorization.'''
  E = 0
  if e < 1.0 :
    M = mod2pi(M); # avoid numerical artefacts for negative numbers
    E = M if e < 0.8 else _np.pi
    F = E - e*_np.sin(E) - M
    for i in range(100):
      E = E - F/(1.-e*_np.cos(E))
      F = E - e*_np.sin(E) - M
      if _np.all(_np.abs(F) < 1.0e-16) : break
    E = mod2pi(E)
    return E
  else:
    E = M/_np.abs(M)*_np.log(2.*_np.abs(M)/e + 1.8)
    F = E - e*_np.sinh(E) + M
    for i in range(100):
      E = E - F/(1.0 - e*_np.cosh(E))
      F = E - e*_np.sinh(E) + M
      if _np.all(_np.abs(F) < 1.0e-16): break
    return E

def E_to_f(e, E):
  '''Converts eccentric anomaly to true anomaly. Implemented from REBOUND using Numpy to handle vectorization.'''
  if e > 1. :return mod2pi(2.*_np.arctan(_np.sqrt((1.+e)/(e-1.))*_np.tanh(0.5*E)))
  else: return mod2pi(2.*_np.arctan(_np.sqrt((1.+e)/(1.-e))*_np.tan(0.5*E)))

def M_to_f(e, M):
  '''Converts mean anomaly to true anomaly. Implemented from REBOUND using Numpy to handle vectorization.'''
  E = M_to_E(e, M)
  return E_to_f(e, E)

############################################################
############### Properties and Elements ####################
############################################################

def calculate_angular_momentum(sim):
    '''
        Calculates the angular momentum of the system and of each particle in the system.

        Args:
          sim (Simulation): The REBOUND Simulation to calculate the angular momentum of.
        
        Returns:
          L (array): The angular momentum of the system and of each particle in the system.
        
        Example:
          ```python
          import rebound
          import airball
          sim = rebound.Simulation()
          sim.add(m=1)
          sim.add(m=5e-5, a=30)
          airball.tools.calculate_angular_momentum(sim) 
          ```
    '''
    L = _np.zeros((sim.N, 3))
    L[0] = sim.angular_momentum()
    for i,p in enumerate(sim.particles[1:]):
        L[i+1,0] = p.m*(p.y*p.vz - p.z*p.vy)
        L[i+1,1] = p.m*(p.z*p.vx - p.x*p.vz)
        L[i+1,2] = p.m*(p.x*p.vy - p.y*p.vx)
    return L

def calculate_eccentricity(sim, star):
  '''
    Calculates the eccentricity of the flyby star.

    Args:
      sim (Simulation): The REBOUND Simulation to calculate the eccentricity with respect to.
      star (Star): The Star to calculate the eccentricity of.
    
    Returns:
      e (float): The eccentricity of the flyby star.
    
    Example:
      ```python
      import rebound
      import airball
      sim = rebound.Simulation()
      sim.add(m=1)
      sim.add(m=5e-5, a=30)
      star = airball.Star(m=1, b=500, v=5)
      airball.tools.calculate_eccentricity(sim, star) 
      ```
  '''
  mu = gravitational_mu(sim, star)
  return vinf_and_b_to_e(mu, star.b, star.v)

def vinf_and_b_to_e(mu, star_b, star_v):
    '''
        Using the impact parameter to convert from the relative velocity at infinity between the two stars to the eccentricity of the flyby star. Equation (2) from [Spurzem et al. (2009)](https://ui.adsabs.harvard.edu/abs/2009ApJ...697..458S/abstract)

        Args:
          mu (Quantity): The total mass of the system (Sun, planets, and flyby star) times the gravitational constant G
          star_b (Quantity): The impact parameter, `b`, of the flyby star. Default units are AU.
          star_v (Quantity): The relative velocity at infinity between the central star and the flyby star (hyperbolic excess velocity). Default units are km/s.
        
        Returns:
          star_e (Quantity): The eccentricity of the flyby star.
    '''

    star_b = verify_unit(star_b, _u.au)
    star_v = verify_unit(star_v, _u.km/_u.s)

    numerator = star_b * star_v**2.
    return _np.sqrt(1 + (numerator/mu)**2.) * _u.dimensionless_unscaled

def vinf_and_q_to_e(mu, star_q, star_v):
    '''
        Using the perihelion to convert from the relative velocity at infinity between the two stars to the eccentricity of the flyby star.

        Args:
          mu (Quantity): The total mass of the system (Sun, planets, and flyby star) times the gravitational constant G
          star_q (Quantity): The perihelion of the flyby star
          star_v (Quantity): The relative velocity at infinity between the central star and the flyby star (hyperbolic excess velocity)
        
        Returns:
          star_e (Quantity): The eccentricity of the flyby star.
    '''

    star_q = verify_unit(star_q, _u.au)
    star_vinf = verify_unit(star_v, _u.km/_u.s)
    return (1 + star_q * star_vinf * star_vinf / mu) * _u.dimensionless_unscaled

def vinf_and_q_to_b(mu, star_q, star_v):
    '''
        Using the perihelion to convert from the relative velocity at infinity between the two stars to the eccentricity of the flyby star.

        Args:
          mu (Quantity): The total mass of the system (Sun, planets, and flyby star) times the gravitational constant G
          star_q (Quantity): The perihelion of the flyby star
          star_v (Quantity): The relative velocity at infinity between the central star and the flyby star (hyperbolic excess velocity)
        
        Returns:
          star_b (Quantity): The impact parameter, `b`, of the flyby star.
    '''

    mu = verify_unit(mu, (_u.au**3)/(_u.yr2pi**2))
    star_q = verify_unit(star_q, _u.au)
    star_vinf = verify_unit(star_v, _u.km/_u.s)
    star_e = 1 + star_q * star_vinf * star_vinf / mu
    return verify_unit(star_q * _np.sqrt((star_e + 1.0)/(star_e - 1.0)), _u.au)

def gravitational_mu(sim, star=None, star_mass=None):
    '''
      Calculate the gravitational parameter, mu, of the system. The gravitational parameter is the total mass of the system (Sun, planets, and flyby star) times the gravitational constant G.

      Args:
        sim (Simulation): The REBOUND Simulation to calculate the gravitational parameter of.
        star (Star): The Star to calculate the gravitational parameter of.
        star_mass (Quantity): The mass of the flyby star to calculate the gravitational parameter of.
      
      Returns:
        mu (Quantity): The gravitational parameter of the system.
    '''
    # Convert the units of the REBOUND Simulation into Astropy Units.
    units = rebound_units(sim)
    G = (sim.G * units.length**3 / units.mass / units.time**2)
    if star is not None and star_mass is not None: raise Exception('Cannot define both star and star_mass.')
    elif star is not None and star_mass is None: star_mass = verify_unit(star.mass, units.mass)
    elif star is None and star_mass is not None: star_mass = verify_unit(star_mass, units.mass)
    else: raise Exception('Either star or star_mass must be defined.')
    return G * (system_mass(sim) * units.mass + star_mass)

def calculate_periastron(sim, star):
    '''
        Using the impact parameter and the relative velocity at infinity between the two stars convert to the periastron of the flyby star.

        Args:
          sim (Simulation): The REBOUND Simulation to calculate the periastron with respect to.
          star (Star): The Star to calculate the periastron of.
        
        Returns:
          star_q (Quantity): The periastron of the flyby star.
    '''
    star_e = calculate_eccentricity(sim, star)
    return star.b * _np.sqrt((star_e - 1.0)/(star_e + 1.0))

def system_mass(sim):
    '''
        The total bound mass of the system. The total bound mass is the mass of the central star plus the mass of all the objects on bound orbits around the central star.

        Args:
          sim (Simulation): The REBOUND Simulation to calculate the system mass of.
        
        Returns:
          total_mass (Quantity): The total bound mass of the system.
    '''
    total_mass = 0
    for i,p in enumerate(sim.particles):
        if i == 0: total_mass += p.m
        elif p.a > 0: total_mass += p.m
        else: pass
    return total_mass      

def semilatus_rectum(**kwargs):
    '''Calculate the semi-latus rectum of a hyperbolic orbit, $l = a(1-e^2)$.'''
    return kwargs['a']*(1.0 - kwargs['e']*kwargs['e'])

def hyperbolic_elements(sim, star, rmax, values_only=False):
    """
    Calculate the flyby star's hyperbolic orbital elements based on the provided Simulation and starting distance (rmax).

    Args:
      sim (Simulation): The simulation with two bodies, a central star and a planet.
      star (Star): The star that is flying by.
      rmax (float): The starting distance of the flyby star. Defaults to units of AU.
      values_only (bool): Whether to return only the values of the hyperbolic orbital elements. If True, then the results can be used to add a new particle to a REBOUND Simulation. Defaults to False.

    Returns:
      elements (dict): A dictionary containing the hyperbolic orbital elements: `{m, a, e, inc, omega, Omega, f, T}`.
      values_only (dict): A dictionary containing the hyperbolic orbital elements: `m`, `a`, `e`, `inc`, `omega`, `Omega`, `f`.

    Raises:
      RuntimeError: If the value for `rmax` is smaller than the impact parameter, `b`.

    Example:
      ```python
      import rebound
      import airball
      sim = rebound.Simulation()
      sim.add(m=1)
      sim.add(m=5e-5, a=30)
      star = airball.Star(m=1, b=500, v=5)
      elements = hyperbolic_elements(sim, star, rmax=100)
      ```
    """
    e = calculate_eccentricity(sim, star)
    a = -star.b/_np.sqrt(e**2. - 1.) # Compute the semi-major axis of the flyby star
    l = semilatus_rectum(a=a, e=e) # Compute the semi-latus rectum of the hyperbolic orbit to get the true anomaly

    rmax = verify_unit(rmax, _u.au)
    if star.N > 1 and not isList(rmax): rmax = _np.array(star.N * [rmax.value]) << rmax.unit
    with _warnings.catch_warnings():
      _warnings.simplefilter("error")
      # Make sure that the value for rmax is sufficiently large.
      div = (_np.divide(l, rmax, out=_np.full_like(rmax, 0), where=rmax!=0) - 1.0)/e # if rmax is 0, then set f=0. Catch divide by zero warning.
      try:
        if star.N > 1:
          if _np.any(rmax[rmax != 0] < star.b[rmax != 0]): raise RuntimeWarning()
        else:
          if rmax < star.b and rmax != 0: raise RuntimeWarning()
        f = _np.where(rmax==0, 0*_u.rad, _np.arccos(div)) # Compute the true anomaly, if rmax is 0, then set f=0.
      except RuntimeWarning as err: 
        if rmax.shape == (): raise RuntimeError(f'{err}, rmax={rmax:1.6g} likely not larger than impact parameter, b={star.b:1.6g}.') from err
        else: raise RuntimeError(f'{err}, rmax={rmax[rmax < star.b][0]:1.6g} likely not larger than impact parameter, b={star.b[rmax < star.b][0]:1.6g}.') from err

    mu = gravitational_mu(sim, star)
    # Compute the time to periapsis from the switching point (-a because the semi-major axis is negative).
    with _u.set_enabled_equivalencies(_u.dimensionless_angles()):
        E = _np.arccosh((_np.cos(f)+e)/(1.+e*_np.cos(f))) # Compute the eccentric anomaly
        M = e * _np.sinh(E)-E # Compute the mean anomaly
    Tperi = M/_np.sqrt(mu/(-a*a*a))

    if values_only: return {'m':star.m.value, 'a':a.value, 'e':e.value, 'inc':star.inc.value, 'omega':star.omega.value, 'Omega':star.Omega.value, 'f':-f.value}
    return {'m':star.m, 'a':a, 'e':e, 'inc':star.inc, 'omega':star.omega, 'Omega':star.Omega, 'f':-f, 'T':Tperi}

def hyperbolic_plane(sim, star):
    '''
      Calculate the plane of the hyperbolic orbit of the flyby star using the position and velocity vectors of the flyby star when the star is a perihelion.
      
      Args:
        sim (Simulation): The simulation with two bodies, a central star and a planet.
        star (Star): The star that is flying by.

      Returns:
        AB (dict): The normalized vectors defining the plane of the hyperbolic orbit. The vectors are `A` and `B` which are unit vectors in the direction of the perihelion and the ascending node, respectively.
    '''
    e = calculate_eccentricity(sim, star)
    
    cO = _np.cos(star.Omega)
    sO = _np.sin(star.Omega)
    co = _np.cos(star.omega)
    so = _np.sin(star.omega)
    ci = _np.cos(star.inc)
    si = _np.sin(star.inc)

    cf = _np.cos(0)
    sf = _np.sin(0)
    A = unit_vector([(cO*(co*cf-so*sf) - sO*(so*cf+co*sf)*ci), (sO*(co*cf-so*sf) + cO*(so*cf+co*sf)*ci), (so*cf+co*sf)*si])
    B = unit_vector([((e+cf)*(-ci*co*sO - cO*so) - sf*(co*cO - ci*so*sO)), ((e+cf)*(ci*co*cO - sO*so)  - sf*(co*sO + ci*so*cO)), ((e+cf)*co*si - sf*si*so)])
    
    return {'A': A, 'B': B}

def cartesian_elements(sim, star, rmax, values_only=False):
    '''
      Returns the Cartesian elements in the Heliocentric frame, based on the total mass of the REBOUND Simulation.
      Implemented from REBOUND using Numpy to handle vectorization.

      Args:
      sim (Simulation): The simulation with two bodies, a central star and a planet.
      star (Star): The star that is flying by.
      rmax (float): The starting distance of the flyby star. Defaults to units of AU.
      values_only (bool): Whether to return only the values of the hyperbolic orbital elements. If True, then the results can be used to add a new particle to a REBOUND Simulation. Defaults to False.

    Returns:
      elements (dict): A dictionary containing the hyperbolic orbital elements: `{m, a, e, inc, omega, Omega, f, T}`.
      values_only (dict): A dictionary containing the hyperbolic orbital elements: `m`, `a`, `e`, `inc`, `omega`, `Omega`, `f`.

    Raises:
      RuntimeError: If the value for `rmax` is smaller than the impact parameter, `b`.

    Example:
      ```python
      import rebound
      import airball
      sim = rebound.Simulation()
      sim.add(m=1)
      sim.add(m=5e-5, a=30)
      star = airball.Star(m=1, b=500, v=5)
      elements = hyperbolic_elements(sim, star, rmax=100)
      ```
    '''
    units = rebound_units(sim)
    G = (sim.G * units.length**3 / units.mass / units.time**2)

    sim.move_to_hel()
    primary = sim.com()
    sim.move_to_com()
    elements = hyperbolic_elements(sim, star, rmax=rmax, values_only=False)
    m,a,e,inc,omega,Omega,f = elements['m'], elements['a'], elements['e'], elements['inc'], elements['omega'], elements['Omega'], elements['f']
    if _np.any(e < 0.): raise ValueError('Eccentricity must be greater than or equal to zero.')
    if _np.any(e > 1.):
        if _np.any(a > 0.):
            raise ValueError('Bound orbit (a > 0) must have e < 1.')
    else:
        if _np.any(a < 0.):
            raise ValueError('Unbound orbit (a < 0) must have e > 1.')
    if _np.any(e*_np.cos(f) < -1.):
        raise ValueError('Unbound orbit can\'t have f set beyond the range allowed by the asymptotes set by the parabola.')
    if primary.m < 1e-15:
        raise ValueError('Primary has no mass.')

    r = a*(1-e*e)/(1 + e*_np.cos(f))
    v0 = _np.sqrt(G*(m + primary.m*units.mass)/a/(1.-e*e))  # in this form it works for elliptical and hyperbolic orbits

    cO = _np.cos(Omega)
    sO = _np.sin(Omega)
    co = _np.cos(omega)
    so = _np.sin(omega)
    cf = _np.cos(f)
    sf = _np.sin(f)
    ci = _np.cos(inc)
    si = _np.sin(inc)

    # Murray & Dermott Eq 2.122
    x = primary.x * units.length + r*(cO*(co*cf-so*sf) - sO*(so*cf+co*sf)*ci)
    y = primary.y * units.length + r*(sO*(co*cf-so*sf) + cO*(so*cf+co*sf)*ci)
    z = primary.z * units.length + r*(so*cf+co*sf)*si

    # Murray & Dermott Eq. 2.36 after applying the 3 rotation matrices from Sec. 2.8 to the velocities in the orbital plane
    vx = primary.vx * units.length/units.time + v0*((e+cf)*(-ci*co*sO - cO*so) - sf*(co*cO - ci*so*sO))
    vy = primary.vy * units.length/units.time + v0*((e+cf)*(ci*co*cO - sO*so)  - sf*(co*sO + ci*so*cO))
    vz = primary.vz * units.length/units.time + v0*((e+cf)*co*si - sf*si*so)

    if values_only: return {'m':m.value, 'x':x.value, 'y':y.value, 'z':z.value, 'vx':vx.value, 'vy':vy.value, 'vz':vz.value}
    return {'m':m, 'x':x, 'y':y, 'z':z, 'vx':vx, 'vy':vy, 'vz':vz, 'T':elements['T']}

def impulse_gradient(star):
    '''Calculate the impulse gradient for a flyby star, $\\frac{2 G M}{v b^2}$.'''
    G = (1 * _u.au**3 / _u.solMass / _u.yr2pi**2)
    return ((2.0 * G * star.m) / (star.v * star.b**2.0)).to(_u.km/_u.s/_u.au)

############################################################
############# Stellar Environment Functions ################
############################################################

def maxwell_boltzmann_dispersion_from_scale(scale):
    '''
        Converts velocity dispersion (variance) $\\sigma$ to scale factor $a$ for [Maxwell-Boltzmann distributions](https://en.wikipedia.org/wiki/Maxwell-Boltzmann_distribution), $\\sigma = a \\sqrt{\\frac{(3\\pi - 8)}{\\pi}}$.
    '''
    return scale * _np.sqrt((3.0*_np.pi - 8.0)/(_np.pi))

def maxwell_boltzmann_scale_from_dispersion(sigma):
    '''
        Converts velocity dispersion (variance) $\\sigma$ to scale factor $a$ for [Maxwell-Boltzmann distributions](https://en.wikipedia.org/wiki/Maxwell-Boltzmann_distribution), $a = \\sqrt{\\frac{\\pi\\sigma^2}{3\\pi - 8}}$.
    '''
    return _np.sqrt((_np.pi*_np.square(sigma))/(3.0*_np.pi - 8.0))

def maxwell_boltzmann_scale_from_mean(mu):
    '''
        Converts mean $\\mu$ to scale factor for [Maxwell-Boltzmann distributions](https://en.wikipedia.org/wiki/Maxwell-Boltzmann_distribution), $a = \\frac{\\mu}{2}\\sqrt{\\frac{\\pi}{2}}$.
    '''
    return _np.sqrt(_np.pi/2.0) * (mu / 2.0)

def maxwell_boltzmann_mean_from_dispersion(sigma):
    '''
        Converts velocity dispersion (variance) $\\sigma$ to mean $\\mu$ for [Maxwell-Boltzmann distributions](https://en.wikipedia.org/wiki/Maxwell-Boltzmann_distribution), $\\mu = 2 \\sqrt{\\frac{2\\sigma^2}{3\\pi - 8}}$.
    '''
    scale = maxwell_boltzmann_scale_from_dispersion(sigma)
    return (2.0 * scale) * _np.sqrt(2.0/_np.pi)

def maxwell_boltzmann_mode_from_dispersion(sigma):
    '''
        Converts velocity dispersion $\\sigma$ to mode (most common or typical value) for [Maxwell-Boltzmann distributions](https://en.wikipedia.org/wiki/Maxwell-Boltzmann_distribution), $\\rm{mode}= \\sqrt{\\frac{2\\pi\\sigma^2}{3\\pi - 8}}$.
    '''
    scale = maxwell_boltzmann_scale_from_dispersion(sigma)
    return scale * _np.sqrt(2.0)

def cross_section(mu, R, v, unit_set=_UnitSet()):
    '''
        The cross-section with gravitational focusing, $σ = πb^2$ considers gravitational focussing where $b = q \\sqrt(1 + \\frac{2GM}{q v_∞^2})$ is the impact parameter, $q$ is the perihelion, $v_∞$ is the relative velocity at infinity, and $M$ is the mass of the flyby star.

        $$\\sigma = \\pi R^2 \\left(1 + \\frac{2GM}{Rv^2}\\right)$$
        
        Args:
          mu (quantity): The mass of flyby star (default units: solMass)
          R (float): The maximum interaction radius (default units: AU)
          v (float): The typical velocity from the distribution (default units: km/s)
          unit_set (airball.units.UnitSet): The set of units to use for the calculation (default [UnitSet][airball.units.UnitSet] units)
    '''

    v = verify_unit(v, unit_set.velocity)
    R = verify_unit(R, unit_set.length)
    mu = verify_unit(mu, unit_set.length**3 / unit_set.time**2)

    return (_np.pi * R**2) * (1 + 2*mu/(R * v**2))

def encounter_rate(n, v, R, M, unit_set=_UnitSet()):
    '''
        The expected flyby encounter rate within an stellar environment,  $\\Gamma = ⟨nσv⟩$
        
        Args:
          n (float): The stellar number density (default units: $\\rm{pc}^{-3}$)
          v (float): The average velocity  (default units: km/s)
          R (float): The interaction radius (default units: AU)
          M (float): The total mass of all the objects in the system such as the Sun, planets, star, etc. (default units: $M_\\odot$)
          unit_set (airball.units.UnitSet): The set of units to use for the calculation (default [UnitSet][airball.units.UnitSet] units)
        
        Returns:
          rate (float): The expected flyby encounter rate within an stellar environment
    '''
    n = verify_unit(n, unit_set.density)
    v = verify_unit(v, unit_set.velocity)
    R = verify_unit(R, unit_set.length)
    M = verify_unit(M, unit_set.mass)

    # Newton's gravitational constant in units of Msun, AU, and Years/2pi (G ~ 1).
    G = (1 * unit_set.length**3 / unit_set.mass / unit_set.time**2)
    sun_mass = 1 * _u.solMass # mass of the Sun in units of Msun

    mu = G * (M + sun_mass) # gravitational parameter of the system

    return n * v * cross_section(mu, R, v, unit_set)


############################################################
################### Units Functions ########################
############################################################

def rebound_units(sim):
    '''Converts the units of a REBOUND Simulation into Astropy Units.
    
    Args:
      sim (Simulation): The REBOUND Simulation to convert the units of.

    Returns:
      simunits (UnitSet): The units of the REBOUND Simulation.

    Example:
      ```python
      import rebound
      import airball
      sim = rebound.Simulation()
      sim.add(m=1)
      sim.add(m=5e-5, a=30)
      airball.tools.rebound_units(sim) # UnitSet with length==au, mass==solMass, and time==yr2pi
      ```
    '''
    defrebunits = {'length': _u.au, 'mass': _u.solMass, 'time': _u.yr2pi}
    simunits = sim.units
    
    for unit in simunits:
        if simunits[unit] == None: simunits[unit] = defrebunits[unit]
        else: simunits[unit] = _u.Unit(simunits[unit])
    return _UnitSet(list(simunits.values()))

def verify_unit(value, unit):
    '''Verifies that the given value has the provided units. If the value is a Quantity and the units are not the same, then the value is converted to the provided units. If the value is not a Quantity, then the value is converted to a Quantity with the provided units. If the value is a numpy array, then the units are applied to each element of the array.'''
    return value.to(unit) if isQuantity(value) else value << unit

def isList(l):
    '''Determines if an object is a list or numpy array.'''
    if isinstance(l,(list,_np.ndarray)): 
       if isinstance(l, _u.quantity.Quantity) and _np.shape(l) == (): return False
       else: return True
    else: return False

def isQuantity(var):
    '''Determines if an object is an Astropy Quantity. Used for Stellar Environment initializations.'''
    return isinstance(var, _u.quantity.Quantity)

