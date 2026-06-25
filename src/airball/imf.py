import functools
import warnings
from copy import deepcopy
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np
from scipy.interpolate import PchipInterpolator

import airball.units as u
import airball.tools as tools


# region MassFunction Protocol
@runtime_checkable
class MassFunction(Protocol):
    """
    A protocol for defining a mass function for use with the `IMF` class.

    The essence of the protocol is to define a callable object that also contains a `unit`
    attribute. Leniency by the `IMF` class is provided if the user does not want to define
    the `unit` attribute. In these cases the IMF will assume that the units of the mass
    function are the same as the `IMF` class.

    The protocol requires:
      - A `unit` attribute (astropy Unit) declaring the mass unit the function expects.
      - A `__call__` method accepting a single argument `x` (float or ndarray, in
        units of `self.unit`) and returning a float or ndarray.

    Note: runtime isinstance() checks only verify the existence of `unit` and
    `__call__`, not the signature. Signature correctness is enforced by calling
    the function with a test value during IMF construction.

    If `unit` is absent, the `IMF` class will assume its own unit and emit a warning.
    If `unit` is present but mismatches the IMF unit, an error is raised.

    Example:
      ```python
      import airball
      import airball.units as u

      A = 0.1
      mf = lambda x: A * x
      mf.unit = u.solMass
      imf = airball.IMF(0.1, 100, mass_function=mf)
      imf.random_mass()
      ```
    """

    unit: object

    def __call__(self, x: float | np.ndarray | u.Quantity) -> float | np.ndarray: ...


# region Available Mass Functions
@dataclass(frozen=True)
class chabrier_2003_single:
    """
    [Chabrier (2003)](https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C/abstract) IMF for single stars, valid for $m \\leq 1\\,M_{\\odot}$.

    The paper defines the IMF in log-space:

    $$\\xi(\\log m) = A \\exp\\left[-\\frac{\\left(\\log_{10}\\left(\\frac{m}{M_\\odot}\\right) - \\log_{10}\\left(\\frac{m_c}{M_\\odot}\\right)\\right)^2 }{ 2 \\sigma^2}\\right]$$

    where:
        - $A = 0.158^{+0.051}_{-0.046}\\,\\log(M_{\\odot})^{-1} \\mathrm{pc}^{-3}$
        - $m_{c} = 0.079^{+0.021}_{-0.016}\\,M_{\\odot}$ (characteristic mass)
        - $\\sigma = 0.69^{+0.05}_{-0.01}$ (dimensionless, width in log space)

    Converting to linear space via the Jacobian
    $\\left|\\frac{\\mathrm{d}(\\log_{10}m)}{\\mathrm{d}m}\\right| = \\frac{1}{m \\ln 10} \\implies \\xi(m) = \\frac{\\xi(\\log m)}{\\left(\\frac{m}{M_\\odot}\\right) \\ln 10}$

    Units of $\\xi(m): M_{\\odot}^{-1} \\mathrm{pc}^{-3}$.

    The paper provides a normalization at $0.7\\,M_{\\odot}$:
        $\\left.\\frac{\\mathrm{d}n}{\\mathrm{d}m}\\right|_{0.7} = 3.8 \\cdot 10^{-2} M_{\\odot}^{-1} \\mathrm{pc}^{-3} \\pm 5\\%$

    Note: when used with the `IMF` class, $A$ and the $\\mathrm{pc}^{-3}$ scaling are normalized
    away during CDF construction and do not affect the sampled mass distribution.

    Args:
        A     (float):    Normalization. Default: $0.158 (\\log M_{\\odot})^{-1} \\mathrm{pc}^{-3}$
        m_c   (float):    Characteristic mass in units of `unit`. Accepts float or Quantity. Default: $0.079\\,M_{\\odot}$
        sigma (float):    Log-space width. Default: $0.69$

    Example:
        ```python
        import airball

        imf = airball.IMF(0.1, 1, mass_function=airball.imf.chabrier_2003_single())
        imf.random_mass()
        ```
    """

    A: float = 0.158
    m_c: float = 0.079
    sigma: float = 0.69

    unit: ClassVar = u.solMass
    _airball_builtin: ClassVar[bool] = True

    def __post_init__(self):
        if isinstance(self.m_c, u.Quantity):
            object.__setattr__(self, "m_c", self.m_c.to(self.unit).value)

    def __call__(self, x: float | np.ndarray | u.Quantity) -> float | np.ndarray:
        m = (x if isinstance(x, u.Quantity) else x * self.unit).to(self.unit)
        m_ratio = (m / self.unit).value  # dimensionless: m/M_☉
        return (
            self.A
            / (m_ratio * np.log(10))  # Jacobian: 1/(m/M_☉ · ln 10)
            * np.exp(-((np.log10(m_ratio) - np.log10(self.m_c)) ** 2) / (2 * self.sigma**2))
        )


@dataclass(frozen=True)
class chabrier_2005_single:
    """
    [Chabrier (2005)](https://ui.adsabs.harvard.edu/abs/2005ASSL..327...41C/abstract) IMF for single stars.

    The paper defines the IMF in log-space with the same lognormal form as Chabrier (2003)
    but with revised constants:

    $$\\xi(\\log m) = A \\exp\\left[-\\frac{\\left(\\log_{10}\\left(\\frac{m}{M_\\odot}\\right) - \\log_{10}\\left(\\frac{m_c}{M_\\odot}\\right)\\right)^2 }{ 2 \\sigma^2}\\right]$$

    where:
        - $A = 0.093\\,\\log(M_{\\odot})^{-1} \\mathrm{pc}^{-3}$
        - $m_{c} = 0.2\\,M_{\\odot}$ (characteristic mass)
        - $\\sigma = 0.55$ (dimensionless, width in log space)

    Converting to linear space via the Jacobian
    $\\left|\\frac{\\mathrm{d}(\\log_{10}m)}{\\mathrm{d}m}\\right| = \\frac{1}{m \\ln 10} \\implies \\xi(m) = \\frac{\\xi(\\log m)}{\\left(\\frac{m}{M_\\odot}\\right) \\ln 10}$

    Units of $\\xi(m): M_{\\odot}^{-1} \\mathrm{pc}^{-3}$.

    Note: when used with the `IMF` class, $A$ and the $\\mathrm{pc}^{-3}$ scaling are normalized
    away during CDF construction and do not affect the sampled mass distribution.

    Args:
        A     (float):    Normalization. Default: $0.093\\,(\\log M_{\\odot})^{-1} \\mathrm{pc}^{-3}$
        m_c   (float):    Characteristic mass in units of `unit`. Accepts float or Quantity. Default: $0.2\\,M_{\\odot}$
        sigma (float):    Log-space width. Default: $0.55$

    Example:
        ```python
        import airball

        imf = airball.IMF(0.01, 1, mass_function=airball.imf.chabrier_2005_single())
        imf.random_mass()
        ```
    """

    A: float = 0.093
    m_c: float = 0.2
    sigma: float = 0.55

    unit: ClassVar = u.solMass
    _airball_builtin: ClassVar[bool] = True

    def __post_init__(self):
        if isinstance(self.m_c, u.Quantity):
            object.__setattr__(self, "m_c", self.m_c.to(self.unit).value)

    def __call__(self, x: float | np.ndarray | u.Quantity) -> float | np.ndarray:
        m = (x if isinstance(x, u.Quantity) else x * self.unit).to(self.unit)
        m_ratio = (m / self.unit).value
        return self.A / (m_ratio * np.log(10)) * np.exp(-((np.log10(m_ratio) - np.log10(self.m_c)) ** 2) / (2 * self.sigma**2))


@dataclass(frozen=True)
class salpeter_1955:
    """
    [Salpeter (1955)](https://ui.adsabs.harvard.edu/abs/1955ApJ...121..161S/abstract) IMF for single stars.

    The paper defines the IMF (in continuous form) as:

    $$\\xi(m) = \\xi_0 \\left(\\frac{m}{M_\\odot}\\right)^{-2.35}$$

    where $\\xi_0 \\approx 0.03\\,\\mathrm{pc}^{-3}\\,M_{\\odot}^{-1}$ is the local stellar density normalization.

    The $m/M_\\odot$ ratio makes this function scale-free with respect to mass units.

    Note: when used with the `IMF` class, $\\xi_0$ is normalized away during CDF
    construction and does not affect the sampled mass distribution. It is
    retained here for scientific fidelity to the original paper.

    Args:
        xi_0 (float): Local stellar density normalization. Default: $0.03\\,\\mathrm{pc}^{-3}\\,M_{\\odot}^{-1}$

    Example:
        ```python
        import airball

        imf = airball.IMF(0.1, 100, mass_function=airball.imf.salpeter_1955())
        imf.random_mass()
        ```
    """

    xi_0: float = 0.03

    unit: ClassVar = u.solMass
    _airball_builtin: ClassVar[bool] = True

    def __call__(self, x: float | np.ndarray | u.Quantity) -> float | np.ndarray:
        m = (x if isinstance(x, u.Quantity) else x * self.unit).to(self.unit)
        m_ratio = (m / self.unit).value  # dimensionless: m/M_☉
        return self.xi_0 * m_ratio**-2.35


@dataclass(frozen=True)
class kroupa_1993:
    """
    [Kroupa, Tout & Gilmore (1993)](https://ui.adsabs.harvard.edu/abs/1993MNRAS.262..545K/abstract) IMF for single stars.

    The IMF is a piecewise power law in linear mass:

    $$\\xi(m) = \\begin{cases} A_1 \\left(\\frac{m}{M_\\odot}\\right)^{-\\alpha_1} & m_1 \\leq \\frac{m}{M_\\odot} < m_2 \\\\ A_2 \\left(\\frac{m}{M_\\odot}\\right)^{-\\alpha_2} & m_2 \\leq \\frac{m}{M_\\odot} < m_3 \\\\ A_3 \\left(\\frac{m}{M_\\odot}\\right)^{-\\alpha_3} & \\frac{m}{M_\\odot} \\geq m_3 \\end{cases}$$

    where continuity is enforced at each break point, so:

    $$A_2 = A_1 \\cdot m_2^{\\,(\\alpha_2 - \\alpha_1)}, \\quad A_3 = A_2 \\cdot m_3^{\\,(\\alpha_3 - \\alpha_2)}$$

    where:
        - $\\alpha_1 = 1.3$, $m_1 = 0.08\\,M_{\\odot}$
        - $\\alpha_2 = 2.2$, $m_2 = 0.5\\,M_{\\odot}$
        - $\\alpha_3 = 2.7$, $m_3 = 1.0\\,M_{\\odot}$
        - $A_1 = 0.035$ (normalization)

    Note: when used with the `IMF` class, $A_1$ is normalized away during CDF
    construction and does not affect the sampled mass distribution.

    Args:
        alpha_1 (float):    Power law index for $m < m_2$. Default: $1.3$
        alpha_2 (float):    Power law index for $m_2 \\leq m < m_3$. Default: $2.2$
        alpha_3 (float):    Power law index for $m \\geq m_3$. Default: $2.7$
        m_1     (float):    Lower break mass in units of `unit`. Accepts float or Quantity. Default: $0.08\\,M_{\\odot}$
        m_2     (float):    First break mass in units of `unit`. Accepts float or Quantity. Default: $0.5\\,M_{\\odot}$
        m_3     (float):    Second break mass in units of `unit`. Accepts float or Quantity. Default: $1.0\\,M_{\\odot}$
        A_1     (float):    Normalization of first segment. Default: $0.035$

    Example:
        ```python
        import airball

        imf = airball.IMF(0.1, 100, mass_function=airball.imf.kroupa_1993())
        imf.random_mass()
        ```
    """

    alpha_1: float = 1.3
    alpha_2: float = 2.2
    alpha_3: float = 2.7
    m_1: float = 0.08
    m_2: float = 0.5
    m_3: float = 1.0
    A_1: float = 0.035
    # Derived from the fields above
    A_2: float = field(init=False, compare=False, hash=False, repr=False, default=0.0)
    A_3: float = field(init=False, compare=False, hash=False, repr=False, default=0.0)

    unit: ClassVar = u.solMass
    _airball_builtin: ClassVar[bool] = True

    def __post_init__(self):
        # Convert Quantity inputs to float in self.unit
        for name in ("m_1", "m_2", "m_3"):
            val = getattr(self, name)
            if isinstance(val, u.Quantity):
                object.__setattr__(self, name, val.to(self.unit).value)
        # Enforce continuity at break points (derived, not dataclass fields)
        A_2 = self.A_1 * self.m_2 ** (self.alpha_2 - self.alpha_1)
        A_3 = A_2 * self.m_3 ** (self.alpha_3 - self.alpha_2)
        object.__setattr__(self, "A_2", A_2)
        object.__setattr__(self, "A_3", A_3)
        print(self.A_1, self.A_2, self.A_3)

    def __call__(self, x: float | np.ndarray | u.Quantity) -> float | np.ndarray:
        m = (x if isinstance(x, u.Quantity) else x * self.unit).to(self.unit)
        m_ratio = (m / self.unit).value  # dimensionless: m/M_☉
        return np.where(
            m_ratio < self.m_2,
            self.A_1 * m_ratio**-self.alpha_1,
            np.where(
                m_ratio < self.m_3,
                self.A_2 * m_ratio**-self.alpha_2,
                self.A_3 * m_ratio**-self.alpha_3,
            ),
        )


@dataclass(frozen=True)
class default_mass_function:
    """
    Default mass function for the `IMF` class.

    A piecewise function combining Chabrier (2003) for $m \\leq 1\\,M_{\\odot}$ and
    Salpeter (1955) for $m > 1\\,M_{\\odot}$, normalized for continuity at $1\\,M_{\\odot}$.

    The continuity constant at the junction is $x_{0} = \\xi_{\\mathrm{Chabrier}}(1\\,M_{\\odot})$ so that
    $\\left.\\xi_{\\mathrm{Salpeter}}(1\\,M_{\\odot})\\right|_{\\xi_{0} = x_{0}} = \\xi_{\\mathrm{Chabrier}}(1\\,M_{\\odot})$.
    The overall scale factor is absorbed by the `IMF` class during normalization and does not affect sampling.

    Example:
        ```python
        import airball

        imf = airball.IMF(0.1, 100, mass_function=airball.imf.default_mass_function())
        imf.random_mass()
        ```
    """

    # Junction point: always 1 M_☉ (class constant, not a per-instance field)
    m_0: ClassVar[float] = 1.0
    # Computed
    chabrier03: chabrier_2003_single = field(
        init=False,
        compare=False,
        hash=False,
        repr=False,
        default_factory=chabrier_2003_single,
    )
    _x_0: float = field(init=False, compare=False, hash=False, repr=False, default=0.0)

    unit: ClassVar = u.solMass
    _airball_builtin: ClassVar[bool] = True

    def __post_init__(self):
        object.__setattr__(self, "_x_0", self.chabrier03(self.m_0))

    def __call__(self, x: float | np.ndarray | u.Quantity) -> float | np.ndarray:
        m = (x if isinstance(x, u.Quantity) else x * self.unit).to(self.unit)
        m_ratio = (m / self.unit).value
        return np.where(
            m_ratio < self.m_0,
            self.chabrier03(m_ratio),
            self._x_0 * m_ratio**-2.35,  # Salpeter, scaled for continuity
        )


# ── Generic / parametric mass functions ───────────────────────────────────────


@dataclass(frozen=True)
class uniform:
    """
    [Uniform](https://en.wikipedia.org/wiki/Continuous_uniform_distribution) mass function.

    A flat probability density — every mass in the IMF range is equally likely.

    $$\\xi(m) = 1$$

    This is scale-free and has no physical constants. The `IMF` class normalizes
    it over $[m_{\\min},\\, m_{\\max}]$ during CDF construction.

    Example:
        ```python
        import airball

        imf = airball.IMF(0.1, 100, mass_function=airball.imf.uniform())
        imf.random_mass()
        ```
    """

    unit: ClassVar = u.solMass
    _airball_builtin: ClassVar[bool] = True

    def __call__(self, x: float | np.ndarray | u.Quantity) -> float | np.ndarray:
        m = (x if isinstance(x, u.Quantity) else x * self.unit).to(self.unit)
        return np.ones_like((m / self.unit).value)


@dataclass(frozen=True)
class power_law:
    """
    Generic [power law](https://en.wikipedia.org/wiki/Power_law) mass function.

    $$\\xi(m) = A \\left(\\frac{m}{M_\\odot}\\right)^{\\alpha}$$

    The $m/M_\\odot$ ratio makes this scale-free with respect to mass units.

    $A$ is normalized away by the `IMF` class during CDF construction.

    Args:
        alpha (float): Power law index.
        A     (float): Normalization factor. Default: $1.0$

    Example:
        ```python
        import airball

        imf = airball.IMF(0.1, 100, mass_function=airball.imf.power_law(alpha=-2.35))
        imf.random_mass()
        ```
    """

    alpha: float
    A: float = 1.0

    unit: ClassVar = u.solMass
    _airball_builtin: ClassVar[bool] = True

    def __call__(self, x: float | np.ndarray | u.Quantity) -> float | np.ndarray:
        m = (x if isinstance(x, u.Quantity) else x * self.unit).to(self.unit)
        m_ratio = (m / self.unit).value
        return self.A * m_ratio**self.alpha


@dataclass(frozen=True)
class broken_power_law:
    """
    Generic [broken power law](https://en.wikipedia.org/wiki/Power_law#Broken_power_law) mass function.

    $$\\xi(m) = \\begin{cases} A \\left(\\frac{m}{M_\\odot}\\right)^{\\alpha} & m < m_0 \\\\ A \\left(\\frac{m_0}{M_\\odot}\\right)^{(\\alpha - \\beta)} \\left(\\frac{m}{M_\\odot}\\right)^{\\beta} & m \\geq m_0 \\end{cases}$$

    Continuity is enforced at $m_0$.

    $A$ is normalized away by the `IMF` class.

    Args:
        alpha (float):    Power law index below $m_0$.
        beta  (float):    Power law index above $m_0$.
        m_0   (float):    Break mass in units of `unit`. Accepts float or Quantity.
        A     (float):    Normalization factor. Default: $1.0$

    Example:
        ```python
        import airball

        mf = airball.imf.broken_power_law(alpha=-1.3, beta=-2.35, m_0=0.5 * u.solMass)
        imf = airball.IMF(0.1, 100, mass_function=mf)
        imf.random_mass()
        ```
    """

    alpha: float
    beta: float
    m_0: float
    A: float = 1.0

    unit: ClassVar = u.solMass
    _airball_builtin: ClassVar[bool] = True

    def __post_init__(self):
        if isinstance(self.m_0, u.Quantity):
            object.__setattr__(self, "m_0", self.m_0.to(self.unit).value)

    def __call__(self, x: float | np.ndarray | u.Quantity) -> float | np.ndarray:
        m = (x if isinstance(x, u.Quantity) else x * self.unit).to(self.unit)
        m_ratio = (m / self.unit).value
        return np.where(
            m_ratio < self.m_0,
            self.A * m_ratio**self.alpha,
            self.A * self.m_0 ** (self.alpha - self.beta) * m_ratio**self.beta,
        )


@dataclass(frozen=True)
class lognormal:
    """
    Generic [lognormal](https://en.wikipedia.org/wiki/Log-normal_distribution) mass function, defined in linear mass space.

    This is the linear-space form of a lognormal (analogous to the Chabrier
    family), including the Jacobian from the log-space definition:

    $$\\xi(m) = \\frac{A}{\\left(\\frac{m}{M_\\odot}\\right) \\ln 10}\\,\\exp\\left[-\\frac{\\left(\\log_{10}\\left(\\frac{m}{M_\\odot}\\right) - \\mu\\right)^2}{2\\sigma^2}\\right]$$

    where $\\mu$ and $\\sigma$ are the mean and width in $\\log_{10}(m/M_\\odot)$ space.

    $A$ is normalized away by the `IMF` class during CDF construction.

    Args:
        mu    (float): Mean in $\\log_{10}(m/M_\\odot)$ space.
        sigma (float): Standard deviation in $\\log_{10}(m/M_\\odot)$ space.
        A     (float): Normalization factor. Default: $1.0$

    Example:
        ```python
        import airball

        mf = airball.imf.lognormal(mu=np.log10(0.3), sigma=0.5)
        imf = airball.IMF(0.1, 100, mass_function=mf)
        imf.random_mass()
        ```
    """

    mu: float
    sigma: float
    A: float = 1.0

    unit: ClassVar = u.solMass
    _airball_builtin: ClassVar[bool] = True

    def __call__(self, x: float | np.ndarray | u.Quantity) -> float | np.ndarray:
        m = (x if isinstance(x, u.Quantity) else x * self.unit).to(self.unit)
        m_ratio = (m / self.unit).value
        return self.A / (m_ratio * np.log(10)) * np.exp(-((np.log10(m_ratio) - self.mu) ** 2) / (2 * self.sigma**2))


@dataclass(frozen=True)
class loguniform:
    """
    [Log-uniform](https://en.wikipedia.org/wiki/Reciprocal_distribution) mass function.

    A distribution that is uniform in log space, equivalent to:

    $$\\xi(m) = \\frac{A}{m/M_\\odot}$$

    This is the $1/m$ reciprocal distribution. It gives equal probability per decade of mass.

    $A$ is normalized away by the `IMF` class during CDF construction.

    Args:
        A (float): Normalization factor. Default: $1.0$

    Example:
        ```python
        import airball

        imf = airball.IMF(0.1, 100, mass_function=airball.imf.loguniform())
        imf.random_mass()
        ```
    """

    A: float = 1.0

    unit: ClassVar = u.solMass
    _airball_builtin: ClassVar[bool] = True

    def __call__(self, x: float | np.ndarray | u.Quantity) -> float | np.ndarray:
        m = (x if isinstance(x, u.Quantity) else x * self.unit).to(self.unit)
        m_ratio = (m / self.unit).value
        return self.A / m_ratio


# region IMF Class
class IMF:
    """
    Initial Mass Function (IMF).

    An empirical function that describes the initial distribution of masses for a population of
    stars during star formation. [(wikipedia)](https://en.wikipedia.org/wiki/Initial_mass_function).

    It generates random masses based on a given mass function (dN/dM) and provides various
    properties and methods for manipulating and analyzing the IMF.

    Args:
      min_mass (u.Quantity): Minimum mass value of the IMF range.
      max_mass (u.Quantity): Maximum mass value of the IMF range.
      mass_function (callable, optional): Mass function to use for the IMF. Default is a piecewise Chabrier 2003 and Salpeter 1955.
      unit (Unit, optional): Unit of mass. Default is solar masses.
      interpolating_points (float, optional): Number of samples to use for interpolating the CDF. Default is 10^5.
      seed (float, optional): Value to seed the random number generator with. Default is None.

    Attributes:
      min_mass (float): Minimum mass value of the IMF range.
      max_mass (float): Maximum mass value of the IMF range.
      median_mass (float): Median mass value of the IMF.
      seed (float): Value to seed the random number generator with.
      interpolating_points (float): Number of samples to use for interpolating the CDF.
      unit (Unit): Unit of mass.
      normalization_factor (float): Normalization factor for the PDF.
      masses (u.Quantity): Mass values logarithmically spanning the IMF range.
      CDF (function): Cumulative distribution function (CDF) of the IMF.
      PDF (function): Normalized probability density function (PDF) of the IMF.
      IMF (function): Initial mass function (IMF) of the IMF.
    """

    def __init__(
        self,
        min_mass: u.Quantity | float,
        max_mass: u.Quantity | float,
        mass_function: Callable | None = None,
        unit: u.Unit = u.solMass,
        interpolating_points: int = int(1e5),
        seed: int | None = None,
    ):
        self._interpolating_samples = int(interpolating_points)
        self._seed = seed
        self.unit = unit if u.isUnit(unit) else u.solMass

        # Convert min_mass and max_mass to specified unit if they are u.Quantity objects, otherwise assume they are already in the correct unit
        self._min_mass = min_mass << self.unit
        self._max_mass = max_mass << self.unit
        if self._min_mass.value <= 0 or self._max_mass.value <= 0:
            raise ValueError("Minimum and maximum mass values must be greater than 0.")
        if self._max_mass <= self._min_mass:
            raise ValueError("Maximum mass value must be greater than minimum mass.")

        # Determine the probability distribution function (PDF) based on the given mass function or default to Chabrier (2003).
        if callable(mass_function) and not isinstance(mass_function, MassFunction):
            if hasattr(mass_function, "unit"):
                if mass_function.unit != self.unit:
                    raise ValueError(
                        f"mass_function unit '{mass_function.unit}' does not match "
                        f"IMF unit '{self.unit}'. Please ensure both use the same unit."
                    )
                _mass_function = mass_function
            else:
                warnings.warn(
                    f"mass_function has no 'unit' attribute. Assuming IMF unit '{self.unit}'. "
                    "Define a unit on your mass function to silence this warning.",
                    UserWarning,
                )
                _mass_function = functools.wraps(mass_function)(lambda x: mass_function(x))
                _mass_function.unit = unit  # ty:ignore[unresolved-attribute]
        elif mass_function is None:
            mass_function = default_mass_function()
            _mass_function = mass_function
        else:
            if hasattr(mass_function, "unit") and mass_function.unit != self.unit:
                raise ValueError(
                    f"mass_function unit '{mass_function.unit}' does not match "
                    f"IMF unit '{self.unit}'. Please ensure both use the same unit."
                )
            _mass_function = mass_function
        if not isinstance(_mass_function, MassFunction):
            raise ValueError("mass_function does not conform to the MassFunction protocol")
        try:
            _ = _mass_function(np.array([1.0]))
        except TypeError as e:
            raise TypeError(f"mass_function could not be called as mass_function(x): {e}") from None
        self.initial_mass_function = _mass_function

        # Recalculate the IMF properties based on the updated parameters
        self._recalculate()

    def _recalculate(self):
        """Initializes the inverse CDF for the IMF to facilitate efficient random mass sampling.

        This method constructs a logarithmically spaced mass grid to accurately resolve multiple
        orders of magnitude. It applies a Jacobian transformation to map the mass density into
        log-space, then numerically integrates this density using a Piecewise Cubic Hermite
        Interpolating Polynomial (PCHIP) to guarantee strict monotonicity.

        After normalizing the integrated CDF, a final PCHIP interpolator is built mapping
        uniform probabilities from [0, 1] back to log-masses. This is stored as `_inv_cdf`
        for subsequent inverse transform sampling.
        """
        grid: np.ndarray = np.geomspace(self.min_mass.value, self.max_mass.value, self._interpolating_samples)
        log_grid: np.ndarray = np.log(grid)

        # Jacobian transformation with `m = grid`
        # Since g(u) = f(m) |dm/du|, and m = e^u, |dm/du| = m.
        g_vals: np.ndarray = self.initial_mass_function(grid) * grid

        # Fits a PCHIP to the log-density.
        g_spline = PchipInterpolator(log_grid, g_vals)

        # Numerically integrate the interpolated density.
        # G(u) represents the indefinite integral.
        G: PchipInterpolator = g_spline.antiderivative()

        # Calculates the definite integral over the entire range [m_min ,m_max].
        self.normalization_factor = G(log_grid[-1]) - G(log_grid[0])

        # Store these for future use in CDF.
        self._G: PchipInterpolator = G
        self._log_min: float = log_grid[0]

        # CDF values at grid points for building the inverse CDF.
        cdf_vals = (G(log_grid) - G(log_grid[0])) / self.normalization_factor
        cdf_vals[0], cdf_vals[-1] = 0.0, 1.0
        self._log_grid = log_grid
        self._inv_cdf = PchipInterpolator(cdf_vals, log_grid, extrapolate=False)

    def cdf(self, x: u.Quantity | np.ndarray) -> np.ndarray:
        """Cumulative distribution function (CDF) of the IMF."""
        min_mass: float = self.min_mass.value
        max_mass: float = self.max_mass.value
        vals = (self._G(np.log(x)) - self._G(self._log_min)) / self.normalization_factor
        clipped: np.ndarray = np.clip(vals, 0.0, 1.0)
        return np.where(x < min_mass, 0.0, np.where(x > max_mass, 1.0, clipped))

    def pdf(self, x: u.Quantity | np.ndarray) -> np.ndarray:
        """Normalized probability density function (PDF) of the IMF."""
        x: np.ndarray = np.asarray(x)
        min_mass: float = self.min_mass.value
        max_mass: float = self.max_mass.value
        vals: np.ndarray = self.initial_mass_function(x) / self.normalization_factor
        return np.where((x >= min_mass) & (x <= max_mass), vals, 0.0)

    def random_mass(self, size: int | tuple[int, ...] = 1, **kwargs) -> u.Quantity | tuple[u.Quantity, ...]:
        """Generates random mass values from the IMF.

        Args:
          size (int or tuple): Number of mass values to generate. If size is a tuple, it is interpreted as array dimensions. Default: 1.

        Keyword Args:
          seed (int): Seed for the random number generator. Default: None.

        Returns:
          masses (float or ndarray): Randomly generated mass value(s) from the IMF.

        Example:
          ```python
          import airball

          imf = airball.IMF(0.1, 100)
          imf.random_mass()
          ```
        """
        size = tuple(int(i) for i in size) if isinstance(size, tuple) else int(size)
        rng = np.random.default_rng(kwargs.get("seed", self.seed))
        masses = np.exp(self._inv_cdf(rng.uniform(size=size))) << self.unit
        if isinstance(size, tuple) or size > 1:
            return masses
        return masses[0]

    @property
    def mean_mass(self):
        # E[m] = ∫ m * pdf(m) dm = ∫ m * g(t) dt in log-space, where t = ln(m)
        log_grid = self._log_grid
        g_vals = (
            self.initial_mass_function(
                np.exp(log_grid)  # the log-mass grid
            )
            * np.exp(log_grid) ** 2
        )  # extra m factor for expectation
        h_spline = PchipInterpolator(log_grid, g_vals / self.normalization_factor)
        return h_spline.integrate(log_grid[0], log_grid[-1]) << self.unit

    @property
    def median_mass(self):
        return np.exp(self._inv_cdf(0.5)) << self.unit

    def masses(self, size, endpoint=True, unitless=True):
        """
        Convenience function for generating an array of mass values logarithmically spanning the IMF range.

        Args:
          size (int): Number of mass values to generate.
          endpoint (bool, optional): Whether to include the max_mass value in the array. Default: True.

        Returns:
          masses (ndarray): numpy array of mass values logarithmically spanning the IMF range.
        """
        ms = np.geomspace(self.min_mass, self.max_mass, int(size), endpoint=endpoint)
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
        value = value.to(self.unit) if tools.isQuantity(value) else value * self.unit
        if value.value <= 0:
            raise ValueError("Cannot have minimum mass value be less than or equal to 0.")
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
        value = value.to(self.unit) if tools.isQuantity(value) else value * self.unit
        if value.value <= 0:
            raise ValueError("Cannot have maximum mass value be less than or equal to 0.")
        if value.value <= self.min_mass.value:
            raise ValueError("Cannot have maximum mass value be less than or equal to minimum mass value.")
        self._max_mass = value
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
    def interpolating_points(self):
        """
        The number of nodes in the log-mass grid used to construct the PCHIP for interpolating the CDF.

        Recalculates the IMF properties when the `interpolating_points` value is updated.

        Args:
          value (int): New number of samples to use for interpolating the CDF.
        """
        return self._interpolating_samples

    @interpolating_points.setter
    def interpolating_points(self, value):
        self._interpolating_samples = int(value)
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

    _CONFIG_ATTRS = (
        "min_mass",
        "max_mass",
        "initial_mass_function",
        "unit",
        "interpolating_points",
        "seed",
    )

    def __eq__(self, other):
        if not isinstance(other, IMF):
            return NotImplemented
        for attr in self._CONFIG_ATTRS:
            v1, v2 = getattr(self, attr), getattr(other, attr)
            if not bool(v1 == v2):
                return False
        return True

    def __hash__(self):
        vals = []
        for attr in self._CONFIG_ATTRS:
            v = getattr(self, attr)
            if tools.isQuantity(v):
                vals.append((v.value, str(v.unit)))
            else:
                try:
                    vals.append(hash(v))
                except TypeError:
                    vals.append(id(v))
        return hash(tuple(vals))

    def summary(self, *, returned=False) -> str | None:
        """
        Prints a compact summary of the current stats of the Stellar Environment object.
        """
        s = f"<{self.__module__}.{type(self).__name__} object at {hex(id(self))}"
        s += f", m= {self.min_mass.value:,.2f}-{self.max_mass.value:,.1f} {self.unit}"
        s += (
            f", IMF= {self.initial_mass_function.__class__.__name__}"
            if self.initial_mass_function.__class__.__module__ == __name__
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
