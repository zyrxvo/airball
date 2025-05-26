"""
# Flybys in `REBOUND`
- A package for implementing flybys in [hannorein/rebound](https://github.com/hannorein/rebound)
"""

__version__ = "v0.9.4"

from .core import (
    add_star_to_sim,
    flyby,
    flybys,
    hybrid_flyby,
    hybrid_flybys,
    successive_flybys,
    concurrent_flybys,
)
from .environments import (
    StellarEnvironment,
    LocalNeighborhood,
    OpenCluster,
    GlobularCluster,
    GalacticBulge,
    GalacticCore,
)
from .analytic import (
    relative_energy_change,
    energy_change_adiabatic_estimate,
    eccentricity_change_adiabatic_estimate,
    inclination_change_adiabatic_estimate,
)
from .imf import IMF
from .stars import Star, Stars
from .units import UnitSet

__all__ = [
    "add_star_to_sim",
    "flyby",
    "flybys",
    "hybrid_flyby",
    "hybrid_flybys",
    "successive_flybys",
    "concurrent_flybys",
    "StellarEnvironment",
    "LocalNeighborhood",
    "OpenCluster",
    "GlobularCluster",
    "GalacticBulge",
    "GalacticCore",
    "relative_energy_change",
    "energy_change_adiabatic_estimate",
    "eccentricity_change_adiabatic_estimate",
    "inclination_change_adiabatic_estimate",
    "IMF",
    "Star",
    "Stars",
    "UnitSet",
]
