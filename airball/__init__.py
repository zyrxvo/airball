# Copyright 2024  Garett Brown
#
# AIRBALL is free software: you can redistribute it and/or modify it under the terms of
# the GNU General Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# AIRBALL is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with airball.
# If not, see http://www.gnu.org/licenses/.
"""AIRBALL: A companion package to REBOUND for simulating stellar flybys."""

import importlib.metadata
from pathlib import Path

# Load tomlib for Python 3.11+ and tomli for Python 3.10.
try:
    import tomllib
except ImportError:
    import tomli as tomllib

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


def get_version() -> str:
    """Get the version of `airball`."""
    try:
        return importlib.metadata.version("airball")
    except importlib.metadata.PackageNotFoundError:
        pyproject = Path(__file__).parents[1] / "pyproject.toml"
        return tomllib.loads(pyproject.read_text())["project"]["version"]


__version__ = get_version()


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
