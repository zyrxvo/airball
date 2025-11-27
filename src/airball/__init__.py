"""# Flybys in `REBOUND`
- A package for implementing flybys in [hannorein/rebound](https://github.com/hannorein/rebound)
"""

__version__ = "v0.9.4"

from .analytic import (
    eccentricity_change_adiabatic_estimate,
    energy_change_adiabatic_estimate,
    inclination_change_adiabatic_estimate,
    relative_energy_change,
)
from .core import (
    add_star_to_sim,
    concurrent_flybys,
    flyby,
    flybys,
    hybrid_flyby,
    hybrid_flybys,
    successive_flybys,
)
from .environments import (
    GalacticBulge,
    GalacticCore,
    GlobularCluster,
    LocalNeighborhood,
    OpenCluster,
    StellarEnvironment,
)
from .imf import IMF
from .stars import Star, Stars
from .units import UnitSet

__all__ = [
    "IMF",
    "GalacticBulge",
    "GalacticCore",
    "GlobularCluster",
    "LocalNeighborhood",
    "OpenCluster",
    "Star",
    "Stars",
    "StellarEnvironment",
    "UnitSet",
    "add_star_to_sim",
    "concurrent_flybys",
    "eccentricity_change_adiabatic_estimate",
    "energy_change_adiabatic_estimate",
    "flyby",
    "flybys",
    "hybrid_flyby",
    "hybrid_flybys",
    "inclination_change_adiabatic_estimate",
    "relative_energy_change",
    "successive_flybys",
]
