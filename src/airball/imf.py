import numpy as np
import types as _types
from copy import deepcopy
from scipy.integrate import quad, cumulative_trapezoid as cumtrapz
from scipy.stats import rv_continuous
from scipy.interpolate import PchipInterpolator
from . import units as _u
from . import tools as _tools


class Distribution:
    """
    Base class for defining a probability distribution function.
    This class is used to wrap a probability distribution function and its parameters into a single object.

    Args:
      mass_function (function): Probability distribution function.
      args (list): List of arguments for the probability distribution function.

    Returns:
      pdf (float or ndarray): Probability density at the given mass value(s).

    Example:
      ```python
      import airball
      mf = airball.imf.Distribution(lambda x, A: A * x, [1])
      imf = airball.IMF(0.1, 100, mass_function=mf)
      imf.random_mass()
      ```

    """

    def __init__(self, mass_function, args, unit):
        self.mass_function = mass_function
        self.params = args
        self.unit = unit

    def __call__(self, x):
        return self.mass_function(x, *self.params)

    def __hash__(self):
        data = []
        for d in sorted(self.__dict__.items()):
            try:
                data.append((d[0], tuple(d[1])))
            except:  # noqa: E722
                data.append(d)
        data = tuple(data)
        return hash(data)

    def __eq__(self, other):
        if isinstance(other, Distribution):
            return (
                self.mass_function == other.mass_function
                and self.params == other.params
            )
        else:
            return NotImplemented


class chabrier_2003_single(Distribution):
    """
    Chabrier 2003 IMF for single stars.
    This function calculates the probability density for a given mass value (x) based on Equation (17) of Chabrier 2003 for an IMF for single stars.

    $$PDF(x) = \\frac{A}{x} \\exp\\left[-\\frac{(\\log_{10}(x) - \\log_{10}(0.079))^2 }{ 2 \\cdot 0.69^2}\\right]$$

    Args:
      A (float, optional): Normalization factor. Default is 0.158.

    Returns:
      pdf (float or ndarray): Probability density at the given mass value(s).

    Example:
      ```python
      import airball
      imf = airball.IMF(0.1, 100, mass_function=airball.imf.chabrier_2003_single(A=0.158))
      imf.random_mass()
      ```
    """

    def __init__(self, A=0.158, unit=_u.solMass):
        x_0 = 0.079 * _u.solMass.to(unit)
        super().__init__(self._chabrier_2003_single, [x_0, A], unit)

    def _chabrier_2003_single(self, x, x_0, A=0.158):
        return (A / x) * np.exp(-((np.log10(x) - np.log10(x_0)) ** 2) / (2 * 0.69**2))


class chabrier_2005_single(Distribution):
    """
    Chabrier 2005 IMF for single stars.
    This function calculates the probability density for a given mass value (x) based on Equation (1) of Chabrier 2005 for an IMF for single stars.

    $$PDF(x) =  \\frac{A}{x} \\exp\\left[-\\frac{(\\log_{10}(x) - \\log_{10}(0.2))^2 }{ 2 \\cdot 0.55^2}\\right]$$

    Args:
      A (float, optional): Normalization factor. Default is 0.093.

    Returns:
      pdf (float or ndarray): Probability density at the given mass value(s).

    Example:
      ```python
      import airball
      imf = airball.IMF(0.01, 100, mass_function=airball.imf.chabrier_2005_single(A=0.093))
      imf.random_mass()
      ```
    """

    def __init__(self, A=0.093, unit=_u.solMass):
        x_0 = 0.2 * _u.solMass.to(unit)
        super().__init__(self._chabrier_2005_single, [x_0, A], unit)

    def _chabrier_2005_single(self, x, x_0, A=0.093):
        return (A / x) * np.exp(-((np.log10(x) - np.log10(x_0)) ** 2) / (2 * 0.55**2))


class salpeter_1955(Distribution):
    """
    Salpeter 1955 IMF for single stars.
    This function calculates the probability density for a given mass value (x) based on the Salpeter 1955 IMF equation.

    $$PDF(x) = A x^{-2.3}$$

    Args:
      A (float): Normalization factor.

    Returns:
      pdf (float or ndarray): Probability density at the given mass value(s).

    Example:
      ```python
      import airball
      imf = airball.IMF(0.1, 100, mass_function=airball.imf.salpeter_1955(A=1))
      imf.random_mass()
      ```
    """

    def __init__(self, A, unit=_u.solMass):
        super().__init__(self._salpeter_1955, [A], unit)

    def _salpeter_1955(self, x, A):
        return A * x**-2.3


class default_mass_function(Distribution):
    """
    Default mass function for an IMF. This function is a piecewise function that uses the IMF for single stars from Chabrier 2003 for masses less than 1 solar mass and the Salpeter 1955 IMF for masses greater than 1 solar mass.

    Args:
      x (float or ndarray): Mass value(s).

    Returns:
      pdf (float or ndarray): Probability density at the given mass value(s).

    Example:
      ```python
      import airball
      imf = airball.IMF(0.1, 100, mass_function=airball.imf.default_mass_function())
      imf.random_mass()
      ```
    """

    def __init__(self, unit=_u.solMass):
        x_0 = (1 * _u.solMass).to(unit).value
        super().__init__(self._default_mass_function, [x_0], unit)

    def _default_mass_function(self, x, x_0):
        chabrier03 = chabrier_2003_single(A=0.158)
        salpeter55 = salpeter_1955(A=chabrier03(1))
        return np.where(x < x_0, chabrier03(x), salpeter55(x))

    def __eq__(self, other):
        if isinstance(other, default_mass_function):
            return True
        return NotImplemented

    def __hash__(self):
        data = []
        for d in sorted(self.__dict__.items()):
            try:
                data.append((d[0], hash(d[1])))
            except:  # noqa: E722
                data.append(d)
        data = tuple(data)
        return hash(data)


class kroupa_1993(Distribution):
    """
    Kroupa et al. (1993) IMF for single stars.
    This function calculates the probability density for a given mass value (x) based on the (Kroupa et al. (1993))[https://ui.adsabs.harvard.edu/abs/1993MNRAS.262..545K/abstract] IMF equation.

    $$PDF(x) = x_0 + \\frac{0.19 x^{1.55} + 0.05 x^{0.6}}{(1-x)^{0.58}}$$

    Args:
      A (float): Normalization factor.

    Returns:
      pdf (float or ndarray): Probability density at the given mass value(s).

    Example:
      ```python
      import airball
      imf = airball.IMF(0.1, 100, mass_function=airball.imf.salpeter_1955(A=1))
      imf.random_mass()
      ```
    """

    def __init__(self, x_0, unit=_u.solMass):
        super().__init__(self._kroupa_1993, [x_0], unit)

    def _kroupa_1993(self, x, x_0):
        return x_0 + (0.19 * x ** (1.55) + 0.05 * x ** (0.6)) / (1 - x) ** (0.58)

    def __eq__(self, other):
        return super().__eq__(other)

    def __hash__(self):
        return super().__hash__()


class uniform(Distribution):
    """
    Uniform IMF.
    This function calculates the probability density for a given mass value (x) based on a uniform IMF.

    $$PDF(x) = 1$$
    """

    def __init__(self, unit=_u.solMass):
        super().__init__(self._uniform, [], unit)

    def _uniform(self, x):
        return x * 0 + 1

    def __eq__(self, other):
        if isinstance(other, uniform):
            return True
        return NotImplemented


class power_law(Distribution):
    """
    Power law IMF.
    This function calculates the probability density for a given mass value (x) based on a power law IMF.

    $$PDF(x) = A x^\\alpha$$

    Args:
      alpha (float): Power law index.
      A (float): Normalization factor.

    Returns:
      pdf (float or ndarray): Probability density at the given mass value(s).
    """

    def __init__(self, alpha, A, unit=_u.solMass):
        super().__init__(self._power_law, [alpha, A], unit)

    def _power_law(self, x, alpha, A):
        return A * x**alpha


class broken_power_law(Distribution):
    """
    Broken power law IMF.
    This function calculates the probability density for a given mass value (x) based on a broken power law IMF.

    $$PDF(x) = \\begin{cases} A x^\\alpha & x < x_0 \\\\ A x_0^{\\beta - \\alpha} x^\\beta & x \\geq x_0 \\end{cases}$$

    Args:
      alpha (float): Power law index for $x < x_0$.
      beta (float): Power law index for $x ≥ x_0$.
      A (float): Normalization factor.
      x_0 (float): Break point between the two power laws.

    Returns:
      pdf (float or ndarray): Probability density at the given mass value(s).
    """

    def __init__(self, alpha, beta, A, x_0, unit=_u.solMass):
        x_0 = _tools.verify_unit(x_0, unit).value
        super().__init__(self._broken_power_law, [alpha, beta, A, x_0], unit)

    def _broken_power_law(self, x, alpha, beta, A, x_0):
        return np.where(x < x_0, A * x**alpha, A * x_0 ** (alpha - beta) * x**beta)


class lognormal(Distribution):
    """
    Lognormal IMF.
    This function calculates the probability density for a given mass value (x) based on a lognormal IMF.

    $$PDF(x) = A \\exp\\left[-\\frac{(\\log_{10}(x) - \\mu)^2 }{ 2 \\sigma^2}\\right]$$

    Args:
      mu (float): Mean of the lognormal distribution.
      sigma (float): Standard deviation of the lognormal distribution.
      A (float): Normalization factor.

    Returns:
      pdf (float or ndarray): Probability density at the given mass value(s).
    """

    def __init__(self, mu, sigma, A, unit=_u.solMass):
        mu = _tools.verify_unit(mu, unit).value
        sigma = _tools.verify_unit(sigma, unit).value
        super().__init__(self._lognormal, [mu, sigma, A], unit)

    def _lognormal(self, x, mu, sigma, A):
        return A * np.exp(-((np.log10(x) - mu) ** 2) / (2 * sigma**2))


class loguniform(Distribution):
    """
    Loguniform IMF.
    This function calculates the probability density for a given mass value (x) based on a loguniform IMF.

    $$PDF(x) = A\\frac{x_0}{x}$$

    Args:
      A (float, optional): Normalization factor.
      x_0 (float, optional): Location to apply the normalization factor.

    Returns:
      pdf (float or ndarray): Probability density at the given mass value(s).
    """

    def __init__(self, A=1, x_0=1, unit=_u.solMass):
        x_0 = _tools.verify_unit(x_0, unit).value
        super().__init__(self._loguniform, [A, x_0], unit)

    def _loguniform(self, x, A, x_0):
        return x_0 * A / x


class _IMFDist(rv_continuous):
    a: int
    b: int

    def _set_imf(self, mass_function, number_samples):
        # Ensure the mass_function is vectorized
        test_x = np.array([self.a, self.b])
        result = mass_function(test_x)
        # If result is not array-like or is scalar, vectorize
        if np.isscalar(result) or (not hasattr(result, "__len__")):
            mass_function = np.vectorize(mass_function)
        self._mass_function = mass_function
        self._norm, _ = quad(mass_function, self.a, self.b)
        self._build_sampler(number_samples)

    def _pdf(self, x, *args):
        return self._mass_function(x) / self._norm

    def _build_sampler(self, n=1000):
        grid = np.geomspace(self.a, self.b, n)
        cdf_vals = np.insert(cumtrapz(self._mass_function(grid), grid), 0, 0)
        cdf_vals /= cdf_vals[-1]
        cdf_vals[0], cdf_vals[-1] = 0.0, 1.0
        log_grid = np.log(grid)
        self._inv_cdf = PchipInterpolator(cdf_vals, log_grid, extrapolate=False)
        self._cdf_spline = PchipInterpolator(log_grid, cdf_vals, extrapolate=False)

    def _rvs(self, *args, size=None, random_state=None):
        u = random_state.uniform(size=size)
        return np.exp(self._inv_cdf(u))


class IMF:
    """
    Class representing an Initial Mass Function (IMF).
    It generates random masses based on a given mass function (dN/dM) and provides various properties and methods for manipulating and analyzing the IMF.

    Args:
      min_mass (float): Minimum mass value of the IMF range.
      max_mass (float): Maximum mass value of the IMF range.
      mass_function (function, optional): Mass function to use for the IMF. Default is a piecewise Chabrier 2003 and Salpeter 1955.
      unit (Unit, optional): Unit of mass. Default is solar masses.
      number_samples (float, optional): Number of samples to use for interpolating the CDF. Default is 5x10^5.
      seed (float, optional): Value to seed the random number generator with. Default is None.

    Attributes:
      min_mass (float): Minimum mass value of the IMF range.
      max_mass (float): Maximum mass value of the IMF range.
      median_mass (float): Median mass value of the IMF.
      seed (float): Value to seed the random number generator with.
      number_samples (float): Number of samples to use for interpolating the CDF.
      unit (Unit): Unit of mass.
      normalization_factor (float): Normalization factor for the PDF.
      masses (ndarray): Mass values logarithmically spanning the IMF range.
      CDF (function): Cumulative distribution function (CDF) of the IMF.
      PDF (function): Normalized probability density function (PDF) of the IMF.
      IMF (function): Initial mass function (IMF) of the IMF.
    """

    def __init__(
        self,
        min_mass,
        max_mass,
        mass_function=None,
        unit=_u.solMass,
        number_samples=5e5,
        seed=None,
    ):
        self._number_samples = int(number_samples)
        self._seed = seed
        self.unit = unit if _u.isUnit(unit) else _u.solMass

        # Convert min_mass and max_mass to specified unit if they are Quantity objects, otherwise assume they are already in the correct unit
        self._min_mass = min_mass << self.unit
        self._max_mass = max_mass << self.unit
        if self._min_mass.value <= 0 or self._max_mass.value <= 0:
            raise ValueError("Minimum and maximum mass values must be greater than 0.")
        if self._max_mass <= self._min_mass:
            raise ValueError("Maximum mass value must be greater than minimum mass.")

        # Determine the probability distribution function (PDF) based on the given mass function or default to Chabrier (2003).
        if mass_function is None:
            mass_function = default_mass_function()
        elif not isinstance(mass_function, (_types.FunctionType, Distribution)):
            raise ValueError(
                "mass_function must be a function or a Distribution object."
            )
        if isinstance(mass_function, Distribution) and mass_function.unit != self.unit:
            raise ValueError("Mass function unit does not match IMF unit.")

        self.initial_mass_function = mass_function

        # Recalculate the IMF properties based on the updated parameters
        self._recalculate()

    def _recalculate(self):
        self._dist = _IMFDist(a=self.min_mass.value, b=self.max_mass.value)
        self._dist._set_imf(self.initial_mass_function, self.number_samples)
        self.normalization_factor = self._dist._norm

    def cdf(self, x):
        return self._dist._cdf_spline(np.log(x))

    def pdf(self, x):
        return self._dist.pdf(x)

    def random_mass(self, size=1, **kwargs):
        self.seed = kwargs.get("seed", self.seed)
        size = tuple(int(i) for i in size) if isinstance(size, tuple) else int(size)
        masses = self._dist.rvs(size=size, random_state=self.seed) << self.unit
        if isinstance(size, tuple) or size > 1:
            return masses
        return masses[0]

    @property
    def mean_mass(self):
        return self._dist.mean() << self.unit

    @property
    def median_mass(self):
        return self._dist.median() << self.unit

    def masses(self, number_samples, endpoint=True, unitless=True):
        """
        Convenience function for generating an array of mass values logarithmically spanning the IMF range.

        Args:
          number_samples (int): Number of mass values to generate.
          endpoint (bool, optional): Whether to include the max_mass value in the array. Default: True.

        Returns:
          masses (ndarray): numpy array of mass values logarithmically spanning the IMF range.
        """
        ms = np.geomspace(
            self.min_mass, self.max_mass, int(number_samples), endpoint=endpoint
        )
        return ms.value if unitless else ms

    @property
    def min_mass(self):
        """
        The minimum mass value of the IMF range.
        Recalculates the IMF properties when the `min_mass` value is updated.

        Args:
          value (float): New minimum mass value. Must be greater than 0. Units are assumed to be the same as the IMF unit.
        """
        return self._min_mass << self.unit

    @min_mass.setter
    def min_mass(self, value):
        value = value.to(self.unit) if _tools.isQuantity(value) else value * self.unit
        if value.value <= 0:
            raise ValueError(
                "Cannot have minimum mass value be less than or equal to 0."
            )
        self._min_mass = value
        self._recalculate()

    @property
    def max_mass(self):
        """
        The maximum mass value of the IMF range.
        Recalculates the IMF properties when the `max_mass` value is updated.

        Args:
          value (float): New maximum mass value. Must be greater than minimum mass value. Units are assumed to be the same as the IMF unit.
        """
        return self._max_mass << self.unit

    @max_mass.setter
    def max_mass(self, value):
        value = value.to(self.unit) if _tools.isQuantity(value) else value * self.unit
        if value.value <= 0:
            raise ValueError(
                "Cannot have maximum mass value be less than or equal to 0."
            )
        if value.value <= self.min_mass:
            raise ValueError(
                "Cannot have maximum mass value be less than or equal to minimum mass value."
            )
        self._max_mass = (
            value.to(self.unit) if _tools.isQuantity(value) else value * self.unit
        )
        self._recalculate()

    @property
    def mass_range(self):
        """
        Median mass value of the IMF.
        """
        return [self.min_mass, self.max_mass] << self.unit

    @property
    def seed(self):
        """
        The seed for the random number generator

        Args:
          value (int or None): New seed for the random number generator (int, or None to turn off).
        """
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value

    @property
    def number_samples(self):
        """
        The number of samples to use for interpolating the CDF.
        Recalculates the IMF properties when the `number_samples` value is updated.

        Args:
          value (int): New number of samples to use for interpolating the CDF.
        """
        return self._number_samples

    @number_samples.setter
    def number_samples(self, value):
        self._number_samples = int(value)
        self._recalculate()

    @property
    def CDF(self):
        return self.cdf

    @property
    def PDF(self):
        return self.pdf

    def imf(self, x):
        """The initial mass function (IMF) of the IMF."""
        return self.initial_mass_function(x)

    @property
    def IMF(self):
        return self.imf

    def copy(self):
        """
        Returns a deep copy of the IMF.
        """
        return deepcopy(self)

    def __eq__(self, other):
        # Overrides the default implementation
        if isinstance(other, IMF):
            attrs = [
                "min_mass",
                "max_mass",
                "initial_mass_function",
                "unit",
                "number_samples",
                "seed",
            ]
            equal = True
            for attr in attrs:
                equal_attribute = getattr(self, attr) == getattr(other, attr)
                if not equal_attribute:
                    if _tools.isQuantity(getattr(self, attr)):
                        equal_attribute = (
                            getattr(self, attr).value == getattr(other, attr).value
                        )
                        equal_attribute = equal_attribute and getattr(
                            self, attr
                        ).unit.is_equivalent(getattr(other, attr).unit)
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
                data.append((d[0], hash(d[1])))
            except TypeError:
                pass
        data = tuple(data)
        return hash(data)

    def summary(self, *, returned=False) -> str | None:
        """
        Prints a compact summary of the current stats of the Stellar Environment object.
        """
        s = f"<{self.__module__}.{type(self).__name__} object at {hex(id(self))}"
        s += f", m= {self.min_mass.value:,.2f}-{self.max_mass.value:,.1f} {self.unit}"
        s += (
            f", IMF= {self.initial_mass_function.__class__.__name__}"
            if isinstance(self.initial_mass_function, Distribution)
            else ", IMF= custom"
        )
        s += ">"
        if returned:
            return s
        else:
            print(s)

    def __str__(self):
        return self.summary(returned=True)

    def __repr__(self):
        return self.summary(returned=True)
