import rebound as _rebound
import numpy as _numpy
import json as _json
from ctypes import c_double, pointer, c_int, byref

from .tools import *

def notNone(a):
    """
    Returns True if array a contains at least one element that is not None. Returns False otherwise.
    """
    return a.count(None) != len(a)

class Particle():
    """
        AIRBALL Particle object.
        
        The Particle class is a convenient way to generate flyby objects used in REBOUND simulations. Each exostar is randomly generated (but are always seeded) and is always generated in heliocentric coordinates.

        To create a random Particle object simply type
        >>> ex = Particle()

        The specific parameters of the object can be defined using keyword arguments. All the current parameters that can be specified are in default REBOUND units (AU, Msun, Yr2Pi), and they are:
            exomass             # The mass of the Particle object (default value is 1 Msun).
            primary_mass        # The combined mass of the system the Particle will flyby (default value is 1 Msun).
            seed                # The random seed used in specifying the location and direction of the flyby (default value is randomly selected and saved).
            b                   # The impact parameter (default value is 100 AU).
            b_min,b_max         # Two arguments that specify a range to uniformly select the impact parameter from.
            periastron          # If the impact parameter is not specified, the periastron of the Particle can be specified instead (default value is 100 AU).
            peri_min,peri_max   # If the impact parameter is not specified, the these two arguments that specify a range to uniformly select the periastron from.
            start_dist          # Similar to the REBOUND simulation exit_max_distance property. This specifies how far away the Particle starts in the simulation, effectively the distance at which the influence of the Particle becomes negligable (default value is 1000 AU).
            vinf                # The hyperbolic excess velocity, or the velocity that the Particle would have at infinity (default value is about 0.0335 AU/Yr2Pi or 1 km/s).
            vinf_units          # If the default hyperbolic excess velocity is changed, units of km/s can be specified for convenience (default value is "None", but "km/s" are available).
            from_file           # This is used for loading an Particle object from a binary file (see below).
        
        To save an Particle object you can use 
        >>> ex.save('filename.bin')

        And then to load the Particle object, simply use
        >>> ex = Exostar.load_from_file('filename.bin')
        
    """
    def __init__(self, exomass=None, primary_mass=None, seed=None, b=None, b_min=None, b_max=None, periastron=None, peri_min=None, peri_max=None, start_dist=None, vinf=None, vinf_units=None, from_file=None, vdir=None, rdir=None):
        if from_file is None:
            # Set up characteristics of the perturbing exostar in a heliocentric frame.
            #   Take the primary_mass to be the combined mass of the host star and all planets.
            if exomass is None:
                self._mass = 1.0
            else:
                self._mass = exomass
            if primary_mass is None:
                self._primary_mass = 1.0
            elif type(primary_mass) is _rebound.simulation.Simulation:
                self._primary_mass = sum([p.m for p in primary_mass.particles])
            else:
                self._primary_mass = primary_mass

            # Set seed for random orientation.
            if seed is None:
                self._seed = int(_numpy.random.randint(0, 2**32 - 1))
            else:
                self._seed = int(seed)
            _numpy.random.seed(self._seed)

            # Set periastron or impact parameter.
            self._peri = None
            self._b = None
            peri_set = [periastron, peri_min, peri_max]
            b_set = [b, b_min, b_max]
            if notNone(peri_set) and notNone(b_set):
                raise AssertionError('You cannot specify the periastron and the impact parameter b.')
            if notNone(peri_set):
                if peri_min is None:
                    if peri_max is None:
                        self._peri = periastron
                    elif periastron is None:
                        raise ValueError('You cannot specify only a maximum periastron. Specify a range with peri_min and peri_max, or specify a specific periastron.')
                    else:
                        raise ValueError('You cannot a specific periastron and a maximum periastron. Specify a range with peri_min and peri_max, or specify a specific periastron.')
                elif peri_max is None:
                    if periastron is None:
                        raise ValueError('You cannot specify only a minimum periastron. Specify a range with peri_min and peri_max, or specify a specific periastron.')
                    else:
                        raise ValueError('You cannot a specific periastron and a minimum periastron. Specify a range with peri_min and peri_max, or specify a specific periastron.')
                elif periastron is None:
                    self._peri = _numpy.random.uniform(peri_min, peri_max)
                else:
                    raise ValueError('You cannot a specific periastron and a periastron range. Specify a range with peri_min and peri_max, or specify a specific periastron.')
            if notNone(b_set):
                if b_min is None:
                    if b_max is None:
                        self._b = b
                    elif b is None:
                        raise ValueError('You cannot specify only a maximum impact parameter. Specify a range with b_min and b_max, or specify a specific impact parameter b.')
                    else:
                        raise ValueError('You cannot a specific impact parameter and a minimum impact parameter. Specify a range with b_min and b_max, or specify a specific impact parameter.')
                elif b_max is None:
                    if b is None:
                        raise ValueError('You cannot specify only a minimum impact parameter. Specify a range with b_min and b_max, or specify a specific impact parameter.')
                    else:
                        raise ValueError('You cannot a specific impact parameter and a minimum impact parameter. Specify a range with b_min and b_max, or specify a specific impact parameter.')
                elif b is None:
                    self._b = _numpy.random.uniform(b_min, b_max)
                else:
                    raise ValueError('You cannot a specific impact parameter and an impact parameter range. Specify a range with b_min and b_max, or specify a specific impact parameter.')
            
            # Set distance between star and exostar.
            d = None
            if self._peri is None and self._b is not None:
                d = self._b
            elif self._b is None and self._peri is not None:
                d = self._peri
            else:
                self._b = 100.
                d = self._b
            
            # Randomly vary the orientation of the exostar at the periapse.
            if vdir is None:
                self._v = self._vec(3)
            else:
                self._v = _numpy.array(vdir)/_numpy.linalg.norm(vdir)
            if rdir is None:
                self._r = _numpy.zeros(3)
                self._r[0:2] = d * self._vec(2)
                self._r = _numpy.dot(self._r, self._R(self._v))
            else:
                self._r = d * _numpy.array(rdir)/_numpy.linalg.norm(rdir)

            # Set set the magnitudes of the position and velocity vectors.
            if vinf is not None:
                if vinf_units is None:
                    self._vinf = vinf
                elif vinf_units == 'km/s':
                    au = 149597870700. # meters in 1 AU
                    yr2pi = (86400.*365.25)/twopi # seconds in 1 Yr2Pi
                    self._vinf = (vinf*1000.)*yr2pi/au # In units of AU/Yr2Pi
                else:
                    raise ValueError('The only other units available are "km/s". Otherwise leave as "None".', )
            else: 
                self._vinf = 0.03357365989646265980 # 1.0 km/s in units of AU/Yr2Pi

            rmag = _numpy.linalg.norm(self._r)
            vmag = _numpy.sqrt( self._vinf**2 + 2. * (self._mass + self._primary_mass) / rmag)
            self._v = self._v * vmag

            # Set the exit distance (how far away before the effects of the exostar are negligable). 
            # TODO: Change the starting distance to be dependent on the mass of the exostar.
            if start_dist is None:
                if self._peri is None:
                    if self._b < 1000.:
                        self._start_dist = 1000.
                    else:
                        self._start_dist = 10.*self._b
                elif self._b is None:
                    if self._peri < 1000.:
                        self._start_dist = 1000.
                    else:
                        self._start_dist = 10.*self._peri
            else:
                self._start_dist = start_dist

            # If the impact parameter or the periastron are less than the starting distance, then move the Particle backwards to the starting distance.
            if self._b is None:
                if self._start_dist < self._peri:
                    raise AssertionError('The periastron cannot be larger than the starting distance.')
                
                # Solve Kepler's Equation for the time it takes to move to the starting position.
                # Assuming Natural Units: G = 1 (Msun = 1, AU = 1, Period = Yr2Pi)
                mu = self._primary_mass + self._mass
                a = -mu/self._vinf**2.
                e = 1. - (rmag/a)
                r = self._start_dist
                cosf = (a * (1 - e**2) - r)/(e*r)
                coshF = (e + cosf)/(1 + e*cosf)
                sinhF = _numpy.sqrt(coshF**2 - 1)
                self._start_time = -_numpy.sqrt(((-a)**3)/mu) * (e * sinhF - _numpy.arcsinh(sinhF) )

                # Solve Kepler's Equation for new position.
                _p = _rebound.Particle(m=self._mass, x=self._r[0], y=self._r[1], z=self._r[2], vx=self._v[0], vy=self._v[1], vz=self._v[2])
                _rebound.clibrebound.reb_whfast_kepler_solver(byref(_rebound.Simulation()), byref(_p), c_double(mu), c_int(0), c_double(self._start_time))

                # Extract the Exostar coordinates in the heliocentric frame.
                self._r = _p.xyz
                self._v = _p.vxyz
            elif self._peri is None:
                if self._start_dist < self._b:
                    raise AssertionError('The impact parameter cannot be larger than the starting distance.')
                alpha = _numpy.sqrt(self._start_dist**2 - _numpy.linalg.norm(self._r)**2)
                v = self._v/_numpy.linalg.norm(self._v)
                self._r = (self._r - alpha*v).tolist()
                self._v = self._v.tolist()
                sim = _rebound.Simulation()
                sim.add(m=self._primary_mass)
                sim.add(self.particle)
                self._start_time = -sim.calculate_orbits()[0].T
            else:
                self._start_time = 0.0
        else:
            self.__dict__.update(from_file)
    # End of Constructor

    def save(self, filename):
        dic_copy = self.__dict__.copy()
        _json.dump(dic_copy, open(filename, "w"), ensure_ascii=False)

    @classmethod
    def load_from_file(cls, filename):
        dic = _json.load(open(filename, "r"))
        return cls(from_file=dic)
    
    @classmethod
    def from_dict(cls, dic):
        return cls(from_file=dic)

    @property
    def particle(self):
        '''Returns a REBOUND particle of the exostar in heliocentric coordinates.'''
        return _rebound.Particle(m=self._mass, x=self._r[0], y=self._r[1], z=self._r[2], vx=self._v[0], vy=self._v[1], vz=self._v[2])
    
    @property
    def primary_mass(self):
        return self._primary_mass

    @property
    def start_time(self):
        '''Returns how long it will take the exostar to travel from it's current location to periastron.'''
        return self._start_time

    @property
    def m(self):
        return self._mass

    @property
    def x(self):
        return self._r[0]

    @property
    def y(self):
        return self._r[1]

    @property
    def z(self):
        return self._r[2]

    @property
    def xyz(self):
        '''Returns the position vector.'''
        return self._r

    @property
    def vx(self):
        return self._v[0]

    @property
    def vy(self):
        return self._v[1]

    @property
    def vz(self):
        return self._v[2]

    @property
    def vxyz(self):
        '''Returns the velocity vector.'''
        return self._v

    @property
    def r(self):
        '''Returns the magnitude of the position vector.'''
        return _numpy.linalg.norm(self._r)

    @property
    def v(self):
        '''Returns the magnitude of the velocity vector.'''
        return _numpy.linalg.norm(self._v)

    @property
    def vinf(self):
        '''Returns the magnitude of the hyperbolic excess velocity, or the velocity that the Particle would have at infinity.'''
        return _numpy.linalg.norm(self._vinf)

    @property
    def pe(self):
        if self._peri is None:
            sim = _rebound.Simulation()
            sim.add(m=self._primary_mass)
            sim.add(self.particle)
            self._peri = sim.particles[1].a*(1 - sim.particles[1].e)
        return self._peri

    @property
    def b(self):
        if self._b is None:
            sim = _rebound.Simulation()
            sim.add(m=self._primary_mass)
            sim.add(self.particle)
            self._b = -sim.particles[1].a * _numpy.sqrt(sim.particles[1].e**2. - 1.)
            # raise ValueError('The impact parameter is not defined.')
        return self._b
    
    @property
    def seed(self):
        return self._seed

    def _vec(self, d):
        '''
            Returns a normalized d-vector.
        '''
        r = 2.
        while r > 1.: 
            out = 2. * _numpy.random.rand(d) - 1.
            r = _numpy.linalg.norm(out)
        return out/r

    def _R(self, v):
        '''
            Given a 3-vector "v", returns a matrix used for rotating other vectors to be perpendicular to "v".
        '''
        c = lambda x: _numpy.cos(x)
        s = lambda x: _numpy.sin(x)
        T = float(_numpy.arcsin(v[1]))
        P = float(_numpy.arcsin(v[0]/c(T)))
        rotation = _numpy.array([[c(P), 0, -s(P)], [-s(T)*s(P), c(T), -s(T)*c(P)], [c(T)*s(P), s(T), c(T)*c(P)]])
        flip = _numpy.array([[1, 0, 0], [0, 1, 0], [0, 0, _numpy.sign(v[2])]])
        return _numpy.dot(rotation,flip)
