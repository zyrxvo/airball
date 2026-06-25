import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import rebound
from scipy.stats import expon, maxwell, uniform

from airball import constants as c
from airball import units as u
from airball.analytic import relative_energy_change
from airball.imf import IMF, chabrier_2003_single, power_law
from airball.stars import Star, Stars
from airball.tools import (
    encounter_rate,
    isQuantity,
    maxwell_boltzmann_mean_from_dispersion,
    maxwell_boltzmann_mode_from_dispersion,
    maxwell_boltzmann_scale_from_dispersion,
    verify_unit,
    vinf_and_b_to_q,
)
from airball.units import UnitSet


class StellarEnvironment:
    """
    This is the AIRBALL StellarEnvironment class.
    It encapsulates the relevant data for a static stellar environment.

    Initializing a StellarEnvironment instance.

    Args:
      stellar_density (float): The stellar density of the environment. Default units are stars/pc^3.
      velocity_dispersion (float): The velocity dispersion in the environment. Default units are km/s.
      lower_mass_limit (float): The lower mass limit for stars in the environment. Default units are solar masses.
      upper_mass_limit (float): The upper mass limit for stars in the environment. Default units are solar masses.
      mass_function (callable, optional): A function that defines the mass distribution of stars. Default is None. If None, the mass function is defined by the Chabrier (2003) IMF for stars with mass < 1 solar mass and the Salpeter (1955) IMF for stars with mass >= 1 solar mass.
      maximum_impact_parameter (float, optional): The maximum impact parameter defining the outer limit of the sphere of influence around a stellar system. If not provided, AIRBALL attempts to estimate a reasonable one. Default is None. Default units are AU.
      name (str, optional): The name of the environment. Default is None.
      UNIT_SYSTEM (list, optional): The unit system used in the environment. Default is an empty list. If not provided, the default unit system assigns 'length': AU, 'time': Myr, 'mass': solar mass, 'angle': radians, 'velocity': km/s, 'object': stars, and 'density': stars/pc^3.
      units (airball.units.UnitSet, optional): The units used in the environment. Default is None.
      object_name (str, optional): The name of the object in the environment. Default is None.
      seed (int, optional): The seed fixing the random star generator. Default is None so it's always random.

    Example:
      ```python
      import airball

      my_env = airball.StellarEnvironment(
          stellar_density=10, velocity_dispersion=20, lower_mass_limit=0.08, upper_mass_limit=8, name="My Environment"
      )
      my_star = my_env.random_star()
      my_env.stats()
      ```

    If a `maximum_impact_parameter` is not given, AIRBALL attempts to estimate a reasonable one.
    The Maximum Impact Parameter is radius defining the outer limit of the sphere of influence around a stellar system.
    There are predefined subclasses for the LocalNeighborhood, a generic OpenCluster, a generic GlobularCluster, and the Milky Way center GalacticBulge and GalacticCore.
    """

    def __init__(
        self,
        filename=None,
        *,
        stellar_density=None,
        velocity_dispersion=None,
        lower_mass_limit=None,
        upper_mass_limit=None,
        mass_function=None,
        maximum_impact_parameter=None,
        name=None,
        UNIT_SYSTEM=None,
        units=None,
        object_name=None,
        seed=None,
        interpolating_points=int(1e5),
    ):
        # Initialize StellarEnvironment from file.
        if filename is not None and isinstance(filename, (str, Path)):
            loaded = StellarEnvironment._load(filename)
            self.__dict__ = loaded.__dict__
            return

        # Check to see if an stars object unit is defined in the given UNIT_SYSTEM and if the user defined a different name for the objects.
        UNIT_SYSTEM = [] if UNIT_SYSTEM is None else UNIT_SYSTEM
        self.units = units if isinstance(units, UnitSet) else UnitSet(UNIT_SYSTEM)
        objectUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(u.stars)]
        if objectUnit == [] and object_name is not None:
            self.units.object = u.def_unit(object_name, u.stars)
        elif objectUnit == [] and object_name is None:
            self.units.object = u.stars
        else:
            self.units.object = objectUnit[0]

        if stellar_density is not None:
            self.density = stellar_density
        else:
            raise AssertionError("Stellar Density must be defined.")
        if velocity_dispersion is not None:
            self.velocity_dispersion = velocity_dispersion
        else:
            raise AssertionError("Velocity Dispersion must be defined.")

        if lower_mass_limit is None:
            raise AssertionError("Lower Mass Limit must be defined.")
        if upper_mass_limit is None:
            raise AssertionError("Upper Mass Limit must be defined.")

        self._upper_mass_limit = (
            upper_mass_limit.to(self.units.mass) if isQuantity(upper_mass_limit) else upper_mass_limit * self.units.mass
        )
        self._lower_mass_limit = (
            lower_mass_limit.to(self.units.mass) if isQuantity(lower_mass_limit) else lower_mass_limit * self.units.mass
        )
        self._IMF = IMF(
            min_mass=self._lower_mass_limit,
            max_mass=self._upper_mass_limit,
            mass_function=mass_function,
            unit=self.units.mass,
            interpolating_points=interpolating_points,
            seed=seed,
        )
        self._median_mass = self.IMF.median_mass
        self.maximum_impact_parameter = maximum_impact_parameter

        self.name = name if name is not None else "Stellar Environment"
        self.short_name = self.name.replace(" ", "")
        self.seed = seed if seed is not None else None

    def random_stars(self, size=1, **kwargs):
        """
        Computes a isotropically random star from a stellar environment.

        Args:
          size (int or tuple): The number of stars to generate. If size is a tuple, it is interpreted as array dimensions. Default: 1.

        Keyword Args:
          include_orientation (bool, optional): If True, the orientation of the star is randomly generated. Otherwise, the orientation of the stars are zero. Default: True.
          maximum_impact_parameter (float, optional): The maximum impact parameter of the star. If None, the maximum impact parameter is estimated. Default: None.
          seed (int, optional): The random seed to use. If None is given then it is random every time. Default: None.

        Returns:
          stars (Star or Stars): A Star object or Stars object (if size > 1) with the randomly generated masses, impact parameters, velocities, and orientations in a heliocentric model.

        Example:
          ```python
          import airball

          my_env = airball.StellarEnvironment(
              stellar_density=10, velocity_dispersion=20, lower_mass_limit=0.08, upper_mass_limit=8, name="My Environment"
          )
          my_stars = my_env.random_stars(10)
          ```
        """
        if isinstance(size, tuple):
            size = tuple([int(i) for i in size])
        else:
            size = int(size)

        include_orientation = kwargs.get("include_orientation", True)
        maximum_impact_parameter = kwargs.get("maximum_impact_parameter")
        self.seed = kwargs.get("seed", self.seed)
        seed = np.random.randint(0, int(2**32 - 6)) if self.seed is None else self.seed

        v = (
            maxwell.rvs(
                scale=maxwell_boltzmann_scale_from_dispersion(self.velocity_dispersion),
                size=size,
                random_state=(seed + 1),
            )
            << self.units.velocity
        )  # Relative velocity of the star at infinity.

        max_impact = maximum_impact_parameter if maximum_impact_parameter is not None else self.maximum_impact_parameter
        b = max_impact * np.sqrt(uniform.rvs(size=size, random_state=(seed + 2)))  # Impact parameter of the star.

        m = self.IMF.random_mass(size=size, seed=(seed + 3))  # Mass of the star.

        zeros = np.zeros(size)
        inc = (
            (2 * np.arcsin(np.sqrt(uniform.rvs(size=size, random_state=(seed + 4))))) << self.units.angle
            if include_orientation
            else zeros
        )
        ω = (
            (uniform.rvs(loc=0, scale=(2.0 * np.pi), size=size, random_state=(seed + 5))) << self.units.angle
            if include_orientation
            else zeros
        )
        Ω = (
            (
                uniform.rvs(
                    loc=-np.pi,
                    scale=(2.0 * np.pi),
                    size=size,
                    random_state=(seed + 6),
                )
            )
            << self.units.angle
            if include_orientation
            else zeros
        )

        if isinstance(size, tuple):
            return Stars(
                m=m,
                b=b,
                v=v,
                inc=inc,
                omega=ω,
                Omega=Ω,
                UNIT_SYSTEM=self.UNIT_SYSTEM,
                environment=self,
            )
        elif size > 1:
            return Stars(
                m=m,
                b=b,
                v=v,
                inc=inc,
                omega=ω,
                Omega=Ω,
                UNIT_SYSTEM=self.UNIT_SYSTEM,
                environment=self,
            )
        else:
            return Star(m, b[0], v[0], inc[0], ω[0], Ω[0], UNIT_SYSTEM=self.UNIT_SYSTEM)

    def random_star(self, size=1, **kwargs) -> Star | Stars:
        # Alias for `random_stars`
        return self.random_stars(size=size, **kwargs)

    def stats(self):
        """
        Prints a summary of the current stats of the Stellar Environment.
        """
        s = self.name
        s += "\n------------------------------------------\n"
        s += "{1} Density:     {0:12.4g} \n".format(
            self.density,
            "Stellar" if self.object_unit.to_string() == u.stars.to_string() else "Object",
        )
        s += "Velocity Scale:      {0:12.4g} \n".format(self.velocity_dispersion)
        s += "Mass Range:            {0:6.4g} - {1:1.4g}\n".format(self.lower_mass_limit.value, self.upper_mass_limit)
        s += "Median Mass:         {0:12.4g} \n".format(self.median_mass)
        s += "Mean Mass:           {0:12.4g} \n".format(self.mean_mass)
        s += "Max Impact Param:    {0:12.4g} \n".format(self.maximum_impact_parameter)
        s += "Encounter Rate:      {0:12.4g} \n".format(self.encounter_rate)
        s += "------------------------------------------"
        print(s)

    def copy(self):
        """
        Returns a deep copy of the current Stellar Environment.
        """
        return deepcopy(self)

    def save(self, filename):
        """
        Save the current instance of the StellarEnvironment class to a file using pickle.

        Args:
          filename (str): The name of the file to save the instance to. The file will be saved in binary format.

        Example:
          ```python
          import airball

          se = airball.OpenCluster()
          se.save("open_cluster.se")
          ```
        """
        if not isinstance(filename, (str, Path)):
            raise ValueError("Filename must be a string or Path.")
        with open(filename, "wb") as pfile:
            pickle.dump(self, pfile, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def _load(cls, filename):
        """
        Load an instance of the StellarEnvironment class from a file using pickle.

        Args:
          filename (str): The name of the file to load the instance from. The file should be in binary format, pickled.

        Returns:
          loaded_stars (StellarEnvironment): The loaded instance of the StellarEnvironment class.

        Example:
          ```python
          import airball

          stars = airball.StellarEnvironment("open_cluster.stars")
          ```
        """
        if not isinstance(filename, (str, Path)):
            raise ValueError("Filename must be a string or Path.")
        with open(filename, "rb") as pfile:
            return pickle.load(pfile)

    def __eq__(self, other):
        # Overrides the default implementation
        if isinstance(other, StellarEnvironment):
            attrs = [
                "density",
                "velocity_dispersion",
                "lower_mass_limit",
                "upper_mass_limit",
                "IMF",
                "maximum_impact_parameter",
                "name",
                "units",
                "object_name",
                "seed",
            ]
            equal = True
            for attr in attrs:
                equal_attribute = getattr(self, attr) == getattr(other, attr)
                if not equal_attribute:
                    if isQuantity(getattr(self, attr)):
                        equal_attribute = getattr(self, attr).value == getattr(other, attr).value
                        equal_attribute = equal_attribute and getattr(self, attr).unit.is_equivalent(getattr(other, attr).unit)
                if not equal_attribute:
                    return False
                equal = equal and equal_attribute
            return equal
        else:
            return NotImplemented

    def __hash__(self):
        # Overrides the default implementation
        data = []
        for d in sorted(self.__dict__.items()):
            try:
                data.append((d[0], tuple(d[1])))
            except Exception:
                data.append(d)
        data = tuple(data)
        return hash(data)

    def summary(self, returned=False):
        """
        Prints a compact summary of the current stats of the Stellar Environment object.
        """
        s = f"<{self.__module__}.{type(self).__name__} object at {hex(id(self))}"
        s += f", n= {self.density:1.4g}"
        s += f", v= {self.velocity_dispersion:,.1f}"
        s += f", m= {self.lower_mass_limit.value:,.2f}-{self.upper_mass_limit.value:,.1f} {self.units['mass']}"
        s += ">"
        if returned:
            return s
        else:
            print(s)

    def __str__(self):
        return self.summary(returned=True)

    def __repr__(self):
        return self.summary(returned=True)

    @property
    def object_unit(self):
        """The unit of the object (star) in the environment."""
        return self.units.object

    @property
    def object_name(self):
        """
        Args:
          value (str): The name of the object (star) in the environment.
        """
        return self.units.object.to_string()

    @object_name.setter
    def object_name(self, value):
        self.units.object = u.def_unit(value, u.stars)

    @property
    def UNIT_SYSTEM(self):
        """
        Args:
          value (list of Units): A list of the units to use for the environment.
        """
        return self.units.UNIT_SYSTEM

    @UNIT_SYSTEM.setter
    def UNIT_SYSTEM(self, UNIT_SYSTEM):
        self.units.UNIT_SYSTEM = UNIT_SYSTEM

    @property
    def median_mass(self):
        """
        The median mass of the environment's initial mass function (IMF).
        """
        return self.IMF.median_mass.to(self.units.mass)

    @property
    def mean_mass(self):
        """
        The mean mass of the environment's initial mass function (IMF).
        """
        return self.IMF.mean_mass.to(self.units.mass)

    @property
    def maximum_impact_parameter(self):
        """
        The largest impact parameter to affect a stellar system in the environment. See the examples in [Adiabatic Tests](../examples/adiabatic-tests.ipynb/#stellarenvironmentmaximum_impact_parameter) for more details.
        """
        return self._maximum_impact_parameter.to(self.units.length)

    @maximum_impact_parameter.setter
    def maximum_impact_parameter(self, value):
        if value is not None:
            self._maximum_impact_parameter = value.to(self.units.length) if isQuantity(value) else value * self.units.length
        else:
            sim = rebound.Simulation()
            sim.add(m=1.0)
            sim.add(m=5.2e-05, a=30.2, e=0.013)  # Use Neptune as a test planet.

            def _f(b):
                return np.abs(
                    relative_energy_change(
                        sim,
                        Stars(
                            m=self.upper_mass_limit,
                            b=b << self.units.length,
                            v=np.sqrt(2.0) * maxwell_boltzmann_mean_from_dispersion(self.velocity_dispersion),
                        ),
                        averaged=True,
                    )
                )

            bs = np.logspace(1, 6, 1000) << u.au
            f_vals = np.asarray(_f(bs))  # monotonically decreasing
            bs_vals = bs.value  # plain float values in au
            # Reverse both arrays so f_vals is ascending, as required by np.interp
            self._maximum_impact_parameter = float(np.interp(1e-16, f_vals[::-1], bs_vals[::-1])) << self.units.length

    @property
    def density(self):
        """
        Args:
          value (Quantity): The number density of the environment. Default units: $\\rm{pc}^{-3}$.
        """
        return self._density.to(self.units.density)

    @density.setter
    def density(self, value):
        if isQuantity(value):
            if value.unit.is_equivalent(u.stars / u.m**3):
                self._density = value.to(self.units.density)
            elif value.unit.is_equivalent(1 / u.m**3):
                self._density = (value * self.units.object).to(self.units.density)
            else:
                raise AssertionError("The given density units are not compatible.")
        else:
            self._density = value * self.units.density

    @property
    def velocity_dispersion(self):
        """
        Args:
          value (Quantity): the velocity dispersion of the environment. Default units: km/s.
        """
        return self._velocity.to(self.units.velocity)

    @velocity_dispersion.setter
    def velocity_dispersion(self, value):
        self._velocity = value.to(self.units.velocity) if isQuantity(value) else value * self.units.velocity

    @property
    def velocity_mean(self):
        """The mean velocity of the environment. Default units: km/s."""
        return maxwell_boltzmann_mean_from_dispersion(self.velocity_dispersion).to(self.units.velocity)

    @property
    def velocity_mode(self):
        """Return the most common velocity of the environment. Default units: km/s."""
        return maxwell_boltzmann_mode_from_dispersion(self.velocity_dispersion).to(self.units.velocity)

    @property
    def velocity_rms(self):
        """Return the root-mean-square velocity of the environment. Default units: km/s."""
        v = maxwell.rvs(
            scale=maxwell_boltzmann_scale_from_dispersion(self.velocity_dispersion),
            size=int(1e6),
        )
        return verify_unit(np.sqrt(np.mean(v**2)), self.units.velocity)

    @property
    def lower_mass_limit(self):
        """
        Args:
          value (Quantity): The lower mass limit of the initial mass function (IMF) of the environment. Default units: $M_\\odot$.
        """
        return self.IMF.min_mass.to(self.units.mass)

    @lower_mass_limit.setter
    def lower_mass_limit(self, value):
        self.IMF.min_mass = value

    @property
    def upper_mass_limit(self):
        """
        Args:
          value (Quantity): The upper mass limit of the initial mass function (IMF) of the environment. Default units: $M_\\odot$.
        """
        return self.IMF.max_mass.to(self.units.mass)

    @upper_mass_limit.setter
    def upper_mass_limit(self, value):
        self.IMF.max_mass = value

    @property
    def IMF(self):
        """
        Args:
          value (IMF): The initial mass function (IMF) of the environment. An `airball.IMF` object.
        """
        return self._IMF

    @IMF.setter
    def IMF(self, value):
        if isinstance(value, IMF):
            self._IMF = IMF(
                value.min_mass,
                value.max_mass,
                value.imf,
                value.unit,
                value.interpolating_points,
                value.seed,
            )
        else:
            raise AssertionError("Initial Mass Function (IMF) must be an airball.IMF object.")

    @property
    def encounter_rate(self):
        """
        Compute the expected flyby encounter rate $\\Gamma = ⟨nσv⟩$ for the stellar environment in units of flybys per year.
        The inverse of the encounter rate will give the average number of years until a flyby.

        The encounter rate is computed using the following parameters:

        - n : stellar number density. Default units: $\\rm{pc}^{-3}$
        - σ : interaction cross section. Default units: $\\rm{AU}^2$
        - v : velocity dispersion. Default units: $\\rm{km/s}$

        The interaction cross section $σ = πb^2$ considers gravitational focussing where $b = q \\sqrt(1 + \\frac{2GM}{q v_∞^2})$ determined by the median mass of the environment, the maximum impact parameter, and the relative velocity at infinity derived from the velocity dispersion.
        """
        total_mass = self.mean_mass + 1.0 * self.units.mass  # Assume a 1 solar mass system experiencing a average mass flyby.
        mu = c.G * total_mass
        q_max = vinf_and_b_to_q(mu, self._maximum_impact_parameter, self.velocity_mean)
        return encounter_rate(
            n=self._density,
            v=self.velocity_mean,
            q=q_max,
            M=total_mass,
            unit_set=self.units,
        ).to(self.units.object / self.units.time)

    def cumulative_encounter_times(self, size):
        """
        Returns the cumulative time from t=0 for when to expect the next flyby encounters.
        This function assumes a Poisson Process and uses an Exponential distribution with the encounter rate.

        Args:
            size (int or tuple): The shape of the returned array. If size is an integer, it is treated as the length of the array. If size is a tuple, it is treated as the shape of the array.

        Returns:
            times (Quantity): An array of cumulative encounter times. The shape of the array is determined by the size parameter.

        Example:
            ```python
            import airball

            my_env = airball.StellarEnvironment(
                stellar_density=10, velocity_dispersion=20, lower_mass_limit=0.08, upper_mass_limit=8, name="My Environment"
            )
            my_env.cumulative_encounter_times(10)  # returns an array of 10 cumulative encounter times.
            ```
        """
        if isinstance(size, tuple):
            size = tuple([int(i) for i in size])
            result = np.cumsum(expon.rvs(scale=1 / self.encounter_rate, size=size), axis=1) << self.units.time
            result -= result[:, 0][:, None]
            return result
        else:
            size = int(size)
            result = np.cumsum(expon.rvs(scale=1 / self.encounter_rate, size=size)) << self.units.time
            result -= result[0]
            return result

    def encounter_times(self, size):
        """
        Returns the time between encounters for when to the expect the next flyby encounters.
        Assumes a Poisson Process and uses an Exponential distribution with the encounter rate.

        Args:
            size (int or tuple): The shape of the returned array. If size is an integer, it is treated as the length of the array. If size is a tuple, it is treated as the shape of the array.

        Returns:
            times (Quantity): An array of encounter times. The shape of the array is determined by the size parameter.

        Example:
            ```python
            import airball

            my_env = airball.StellarEnvironment(
                stellar_density=10, velocity_dispersion=20, lower_mass_limit=0.08, upper_mass_limit=8, name="My Environment"
            )
            my_env.encounter_times(10)  # returns an array of 10 encounter times.
            ```
        """
        if isinstance(size, tuple):
            size = tuple([int(i) for i in size])
            return expon.rvs(scale=1 / self.encounter_rate, size=size) << self.units.time
        else:
            size = int(size)
            return expon.rvs(scale=1 / self.encounter_rate, size=size) << self.units.time

    def time_to_next_encounter(self):
        """
        Draw a time to the next expected flyby encounter.
        Assumes a Poisson Process and uses an Exponential distribution with the encounter rate.

        Returns:
            times (Quantity): The next encounter time.

        Example:
            ```python
            import airball

            my_env = airball.StellarEnvironment(
                stellar_density=10, velocity_dispersion=20, lower_mass_limit=0.08, upper_mass_limit=8, name="My Environment"
            )
            my_env.time_to_next_encounter()
            ```
        """
        return expon.rvs(scale=1 / self.encounter_rate) * self.units.time


class LocalNeighborhood(StellarEnvironment):
    """
    This is a `StellarEnvironment` subclass for the Local Neighborhood.
    It encapsulates the relevant data for a static stellar environment representing the local neighborhood of the solar system.

    The stellar density is 0.14 $\\rm{pc}^{-3}$ defined by [Bovy (2017)](https://ui.adsabs.harvard.edu/abs/2017MNRAS.470.1360B/abstract).
    The velocity distribution is defined using a Maxwell-Boltzmann distribution where the velocity dispersion is 20 km/s, defined by [Binnery & Tremaine (2008)](https://ui.adsabs.harvard.edu/abs/2008gady.book.....B/abstract) where the $v_\\rm{rms} \\sim 50$ km/s and [Bailer-Jones et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018A%26A...616A..37B/abstract) so that 90% of stars have v < 100 km/s with an encounter rate of ~20 stars/Myr within 1 pc. However, a more accurate representation of the velocity distribution in the solar neighborhood is a triaxial Gaussian distribution, but that has not been implemented here.
    The mass limits is defined to between 0.08-8 solar masses using Equation (17) from [Chabrier (2003)](https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C/abstract) for single stars when m < 1 and a power-law model from [Bovy (2017)](https://ui.adsabs.harvard.edu/abs/2017MNRAS.470.1360B/abstract) for stars m ≥ 1 to account for depleted stars due to stellar evolution.

    Example:
      ```python
      import airball

      my_local = airball.LocalNeighborhood()
      my_10stars = my_local.random_stars(
          size=10
      )  # returns a Stars object with the masses, impact parameters, velocities, and orientation of the 10 Star objects in a heliocentric model.
      ```
    """

    short_name = "Local"

    _ch03 = chabrier_2003_single(1)

    def local_mass_function(
        x: float | np.ndarray,
        _chabrier03=_ch03,
        _lplaw=power_law(-4.7, float(_ch03(1))),
    ) -> float | np.ndarray:
        """
        This defined using Equation (17) from [Chabrier (2003)](https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C/abstract) for single stars when $m < 1$ and a power-law model from [Bovy (2017)](https://ui.adsabs.harvard.edu/abs/2017MNRAS.470.1360B/abstract) for stars $m \\ge 1$ to account for depleted stars due to stellar evolution.
        """
        return np.where(x > 1, _lplaw(x), _chabrier03(x))

    del _ch03
    local_mass_function.unit = u.solMass

    def __init__(
        self,
        stellar_density=0.14 * u.stars / u.pc**3,
        velocity_dispersion=20.8 * u.km / u.s,
        lower_mass_limit=0.08 * u.solMass,
        upper_mass_limit=8 * u.solMass,
        mass_function=local_mass_function,
        maximum_impact_parameter=10000 * u.au,
        UNIT_SYSTEM=None,
        units=None,
        name="Local Neighborhood",
        object_name=None,
        seed=None,
        interpolating_points=int(1e5),
    ):
        super().__init__(
            stellar_density=stellar_density,
            velocity_dispersion=velocity_dispersion,
            lower_mass_limit=lower_mass_limit,
            upper_mass_limit=upper_mass_limit,
            mass_function=mass_function,
            maximum_impact_parameter=maximum_impact_parameter,
            UNIT_SYSTEM=UNIT_SYSTEM,
            units=units,
            name=name,
            object_name=object_name,
            seed=seed,
            interpolating_points=interpolating_points,
        )


class OpenCluster(StellarEnvironment):
    """
    This is a StellarEnvironment subclass for a generic Open Cluster.
    It encapsulates the relevant data for a static stellar environment representing a generic open cluster.

    The stellar density is 100 $\\rm{pc}^{-3}$ informed by [Adams (2010)](https://ui.adsabs.harvard.edu/abs/2010ARA%26A..48...47A/abstract).
    The velocity scale is 1 km/s informed by [Adams (2010)](https://ui.adsabs.harvard.edu/abs/2010ARA%26A..48...47A/abstract) and [Malmberg, Davies, & Heggie (2011)](https://ui.adsabs.harvard.edu/abs/2011MNRAS.411..859M/abstract).
    The mass limit is defined to between 0.08-100 solar masses using Equation (17) from [Chabrier (2003)](https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C/abstract) for single stars when m < 1 and [Salpeter (1955)](https://ui.adsabs.harvard.edu/abs/1955ApJ...121..161S/abstract) for stars m ≥ 1.

    Example:
      ```python
      import airball

      my_open = airball.OpenCluster()
      my_10stars = my_open.random_stars(
          size=10
      )  # returns a Stars object with the masses, impact parameters, velocities, and orientation of the 10 Star objects in a heliocentric model.
      ```
    """

    short_name = "Open"

    def __init__(
        self,
        stellar_density=100 * u.stars * u.pc**-3,
        velocity_dispersion=1 * u.km / u.s,
        lower_mass_limit=0.08 * u.solMass,
        upper_mass_limit=100 * u.solMass,
        mass_function=None,
        maximum_impact_parameter=1000 * u.au,
        UNIT_SYSTEM=None,
        units=None,
        name="Open Cluster",
        object_name=None,
        seed=None,
        interpolating_points=int(1e5),
    ):
        super().__init__(
            stellar_density=stellar_density,
            velocity_dispersion=velocity_dispersion,
            lower_mass_limit=lower_mass_limit,
            upper_mass_limit=upper_mass_limit,
            mass_function=mass_function,
            maximum_impact_parameter=maximum_impact_parameter,
            UNIT_SYSTEM=UNIT_SYSTEM,
            units=units,
            name=name,
            object_name=object_name,
            seed=seed,
            interpolating_points=interpolating_points,
        )


class GlobularCluster(StellarEnvironment):
    """
    This is a StellarEnvironment subclass for a generic Globular Cluster.
    It encapsulates the relevant data for a static stellar environment representing a generic globular cluster.

    The stellar density is 1000 $\\rm{pc}^{-3}$.
    The velocity scale is 10 km/s.
    The mass limit is defined to between 0.08-1 solar masses using Equation (17) from [Chabrier (2003)](https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C/abstract) for single stars when m < 1. It is assumed that there are no stellar masses greater than 1 solar mass in a globular cluster due to stellar evolution.

    Example:
      ```python
      import airball

      my_glob = airball.GlobularCluster()
      my_10stars = my_glob.random_stars(
          size=10
      )  # returns a Stars object with the masses, impact parameters, velocities, and orientation of the 10 Star objects in a heliocentric model.
      ```
    """

    short_name = "Globular"

    def __init__(
        self,
        stellar_density=1000 * u.stars * u.pc**-3,
        velocity_dispersion=10 * u.km / u.s,
        lower_mass_limit=0.08 * u.solMass,
        upper_mass_limit=1 * u.solMass,
        mass_function=None,
        maximum_impact_parameter=5000 * u.au,
        UNIT_SYSTEM=None,
        units=None,
        name="Globular Cluster",
        object_name=None,
        seed=None,
        interpolating_points=int(1e5),
    ):
        super().__init__(
            stellar_density=stellar_density,
            velocity_dispersion=velocity_dispersion,
            lower_mass_limit=lower_mass_limit,
            upper_mass_limit=upper_mass_limit,
            mass_function=mass_function,
            maximum_impact_parameter=maximum_impact_parameter,
            UNIT_SYSTEM=UNIT_SYSTEM,
            units=units,
            name=name,
            object_name=object_name,
            seed=seed,
            interpolating_points=interpolating_points,
        )


class GalacticBulge(StellarEnvironment):
    """
    This is a StellarEnvironment subclass for a generic Galactic Bulge.
    It encapsulates the relevant data for a static stellar environment representing a generic galactic bulge. This region of the galaxy is more dense than the typical field stars found in spiral arms and has a higher velocity dispersion.

    The stellar density is 50 $\\rm{pc}^{-3}$.
    The velocity scale is 120 km/s.
    The mass limit is defined to between 0.08-10 solar masses using Equation (17) from [Chabrier (2003)](https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C/abstract) for single stars when m < 1 and [Salpeter (1955)](https://ui.adsabs.harvard.edu/abs/1955ApJ...121..161S/abstract) for stars m ≥ 1.

    Example:
      ```python
      import airball

      my_bulge = airball.GalacticBulge()
      my_10stars = my_bulge.random_stars(
          size=10
      )  # returns a Stars object with the masses, impact parameters, velocities, and orientation of the 10 Star objects in a heliocentric model.
      ```
    """

    short_name = "Bulge"

    def __init__(
        self,
        stellar_density=50 * u.stars * u.pc**-3,
        velocity_dispersion=120 * u.km / u.s,
        lower_mass_limit=0.08 * u.solMass,
        upper_mass_limit=10 * u.solMass,
        mass_function=None,
        maximum_impact_parameter=50000 * u.au,
        UNIT_SYSTEM=None,
        units=None,
        name="Milky Way Bulge",
        object_name=None,
        seed=None,
        interpolating_points=int(1e5),
    ):
        super().__init__(
            stellar_density=stellar_density,
            velocity_dispersion=velocity_dispersion,
            lower_mass_limit=lower_mass_limit,
            upper_mass_limit=upper_mass_limit,
            mass_function=mass_function,
            maximum_impact_parameter=maximum_impact_parameter,
            UNIT_SYSTEM=UNIT_SYSTEM,
            units=units,
            name=name,
            object_name=object_name,
            seed=seed,
            interpolating_points=interpolating_points,
        )


class GalacticCore(StellarEnvironment):
    """
    This is a StellarEnvironment subclass for a generic Galactic Core.
    It encapsulates the relevant data for a static stellar environment representing a generic galactic core. This is the densest region of the galaxy and has the highest velocity dispersion.

    The stellar density is $10^4$ $\\rm{pc}^{-3}$.
    The velocity scale is 170 km/s.
    The mass limit is defined to between 0.08-10 solar masses using Equation (17) from [Chabrier (2003)](https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C/abstract) for single stars when m < 1 and [Salpeter (1955)](https://ui.adsabs.harvard.edu/abs/1955ApJ...121..161S/abstract) for stars m ≥ 1.

    Example:
      ```python
      import airball

      my_core = airball.GalacticCore()
      my_10stars = my_core.random_stars(
          size=10
      )  # returns a Stars object with the masses, impact parameters, velocities, and orientation of the 10 Star objects in a heliocentric model.
      ```
    """

    short_name = "Core"

    def __init__(
        self,
        stellar_density=10000 * u.stars * u.pc**-3,
        velocity_dispersion=170 * u.km / u.s,
        lower_mass_limit=0.08 * u.solMass,
        upper_mass_limit=10 * u.solMass,
        mass_function=None,
        maximum_impact_parameter=50000 * u.au,
        UNIT_SYSTEM=None,
        units=None,
        name="Milky Way Core",
        object_name=None,
        seed=None,
        interpolating_points=int(1e5),
    ):
        UNIT_SYSTEM = [u.yr] if UNIT_SYSTEM is None else UNIT_SYSTEM
        super().__init__(
            stellar_density=stellar_density,
            velocity_dispersion=velocity_dispersion,
            lower_mass_limit=lower_mass_limit,
            upper_mass_limit=upper_mass_limit,
            mass_function=mass_function,
            maximum_impact_parameter=maximum_impact_parameter,
            UNIT_SYSTEM=UNIT_SYSTEM,
            units=units,
            name=name,
            object_name=object_name,
            seed=seed,
            interpolating_points=interpolating_points,
        )
