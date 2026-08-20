# Copyright 2024 Garett Brown
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
"""Astropy Units for `airball`."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import astropy.units as _u
from astropy.units import *  # ruff: ignore[undefined-local-with-import-star]  # ty: ignore[invalid-declaration]
from astropy.units.core import UnitBase

if TYPE_CHECKING:
    import builtins
    from collections.abc import ItemsView, Iterator, KeysView, ValuesView

    from astropy.units import Quantity

twopi = math.tau
yrtwopi = _u.def_unit("yrtwopi", _u.yr / twopi, format={"latex": r"(yr/2\pi)"})
yr2pi = _u.def_unit("yr2pi", _u.yr / twopi, format={"latex": r"(yr/2\pi)"})
stars = _u.def_unit("stars")
_u.add_enabled_units([yr2pi, yrtwopi])
_u.add_enabled_aliases({"msun": _u.solMass})


def is_unit(var: builtins.object) -> bool:
    """Determine if an object is an Astropy Unit type."""
    return isinstance(var, UnitBase)


# Backwards compatibility alias.
isUnit = is_unit  # ruff: ignore[mixed-case-variable-in-global-scope]


class UnitSet:
    """Manage the units of another class.

    The default units are `[u.au, u.yr2pi, u.solMass, u.rad, u.km/u.s, u.stars/u.pc**3, u.stars]`.
    If a list of units is provided, the UnitSet will attempt to determine the units of length, time, mass, angle, velocity, density, and object. Thus, if a unit of length and a unit of time are given, but no unit of velocity, then UnitSet will calculate the unit of velocity from the given units of length and time. If no units are provided, the default units will be used.
    Two UnitSets are considered equal if the string representations of the units in each UnitSets are identical.

    Args:
      unit_system (list): A list of Astropy Units describing the units of the system.

    Attributes:
      bases (list[UnitBase]): A list of Astropy Units describing the units of the system.
      units (dict[str, UnitBase]): A dictionary of Astropy Units describing the units of the system. Can also access the dictionary from the object itself.
      length (UnitBase): The unit of length.
      time (UnitBase): The unit of time.
      mass (UnitBase): The unit of mass.
      angle (UnitBase): The unit of angle.
      velocity (UnitBase): The unit of velocity in length/time.
      density (UnitBase): The unit of density in object/length**3.
      object (UnitBase): The unit of an object (such as a star).

    Example:
      ```python
      import airball
      import airball.units as u

      us1 = u.UnitSet([u.pc, u.Myr])
      us2 = u.UnitSet()
      print(us1 == us2)  # False
      print(us1.velocity)  # pc/Myr
      print(us2["velocity"])  # km/s
      ```

    """

    __slots__ = ("_bases", "_units")

    def __init__(self, unit_system: UnitSet | list[UnitBase] | None = None) -> None:
        # Set the default units.
        self._units: dict[str, UnitBase] = {
            "length": _u.au,
            "time": _u.Myr,
            "mass": _u.solMass,
            "angle": _u.rad,
            "velocity": _u.km / _u.s,
            "object": stars,
            "density": stars / _u.pc**3,
        }
        if unit_system is None:
            unit_system = []
        if isinstance(unit_system, list):
            self.unit_system = unit_system
        elif isinstance(unit_system, UnitSet):
            self.unit_system = unit_system.unit_system
        else:
            message: str = "unit_system must be a list of Astropy Units."
            raise TypeError(message)

    def find(self, unit: UnitBase) -> str | None:
        """Return the category key for a unit, or None if not found."""
        return next((k for k, v in self.items() if v.is_equivalent(unit)), None)

    def decompose(self, quantity: Quantity) -> Quantity:
        """Decompose a `Quantity` with units into irreducible units."""
        return quantity.decompose(self.bases)

    @property
    def units(self) -> dict[str, UnitBase]:
        """The dictionary of units for the system."""
        return self._units

    @property
    def unit_system(self) -> list[UnitBase]:
        """The unit system used by Astropy.Units for decomposing."""
        return self._bases

    @property
    def bases(self) -> list[UnitBase]:
        """The unit system used by Astropy.Units for decomposing."""
        return self._bases

    def __getitem__(self, key: str) -> UnitBase:
        """Get a unit by its name."""
        if isinstance(key, str):
            return self.units[key]
        msg = f"Key must be a string, got {type(key).__name__}"
        raise TypeError(msg)

    def __setitem__(self, key: str, value: UnitBase) -> None:
        """Set a unit by its name."""
        if not isinstance(key, str):
            msg = f"Key must be a string, got {type(key).__name__}"
            raise TypeError(msg)
        if key not in self.units:
            msg = f"Unknown unit key {key!r}, must be one of: {', '.join(self.units)}"
            raise KeyError(msg)
        if not isinstance(value, UnitBase):
            msg = f"Value must be a valid Astropy Unit, got {type(value).__name__}"
            raise TypeError(msg)
        self.units[key] = value

    def __str__(self) -> str:
        """Return a human-readable string representation of the UnitSet."""
        parts = [f"{key}: {self.units[key].to_string()}" for key in self.units]
        return "{" + ", ".join(parts) + "}"

    def __repr__(self) -> str:
        """Return a detailed string representation of the UnitSet."""
        return "{\n" + "".join([f"  {k}: {v.to_string()},\n" for k, v in self.items()]) + "}"

    def __iter__(self) -> Iterator[str]:
        """Iterate over the unit keys in the UnitSet."""
        yield from self.units

    def __eq__(self, other: builtins.object) -> bool:
        """Check equality by comparing string representations of all units."""
        if isinstance(other, UnitSet):
            if len(self._units) != len(other._units):
                return False
            return all(u1.to_string() == u2.to_string() for u1, u2 in zip(self.values(), other.values(), strict=True))
        return NotImplemented

    def __hash__(self) -> int:
        """Return a hash of the UnitSet based on its attributes."""
        return hash(tuple(u.to_string() for u in self.values()))

    def keys(self) -> KeysView[str]:
        """Return the unit keys of the UnitSet."""
        return self.units.keys()

    def values(self) -> ValuesView[UnitBase]:
        """Return the unit values of the UnitSet."""
        return self.units.values()

    def items(self) -> ItemsView[str, UnitBase]:
        """Return the unit key-value pairs of the UnitSet."""
        return self.units.items()

    @property
    def length(self) -> UnitBase:
        """The unit of length."""
        return self._units["length"]

    @length.setter
    def length(self, value: UnitBase) -> None:
        """Set the unit of length and update dependent units."""
        self.unit_system = [value]

    @property
    def time(self) -> UnitBase:
        """The unit of time."""
        return self._units["time"]

    @time.setter
    def time(self, value: UnitBase) -> None:
        """Set the unit of time and update dependent units."""
        self.unit_system = [value]

    @property
    def mass(self) -> UnitBase:
        """The unit of mass."""
        return self._units["mass"]

    @mass.setter
    def mass(self, value: UnitBase) -> None:
        """Set the unit of mass."""
        self.unit_system = [value]

    @property
    def angle(self) -> UnitBase:
        """The unit of angle."""
        return self._units["angle"]

    @angle.setter
    def angle(self, value: UnitBase) -> None:
        """Set the unit of angle."""
        self.unit_system = [value]

    @property
    def velocity(self) -> UnitBase:
        """The unit of velocity."""
        return self._units["velocity"]

    @velocity.setter
    def velocity(self, value: UnitBase) -> None:
        """Set the unit of velocity."""
        self.unit_system = [value]

    @property
    def density(self) -> UnitBase:
        """The unit of number density."""
        return self._units["density"]

    @density.setter
    def density(self, value: UnitBase) -> None:
        """Set the unit of number density."""
        self.unit_system = [value]

    @property
    def object(self) -> UnitBase:
        """The unit of an object (e.g., stars)."""
        return self._units["object"]

    @object.setter  # ruff: ignore[builtin-attribute-shadowing]
    def object(self, value: UnitBase) -> None:
        """Set the unit of an object (e.g., stars)."""
        self.unit_system = [value]

    @unit_system.setter
    def unit_system(self, unit_system: list[UnitBase]) -> None:  # ruff: ignore[complex-structure]
        """Set the unit system, inferring missing units from the provided ones."""
        if not unit_system:
            self._bases = list(self._units.values())
            return

        if not all(isinstance(item, UnitBase) for item in unit_system):
            bad = next(item for item in unit_system if not isinstance(item, UnitBase))
            msg = f"All items in unit_system must be Astropy Units, got {type(bad).__name__}: {bad!r}"
            raise TypeError(msg)

        def _find(ref: UnitBase) -> UnitBase | None:
            return next((u for u in unit_system if u.is_equivalent(ref)), None)

        # Assign any explicitly provided units.
        for key, reference in [("length", _u.m), ("time", _u.s), ("mass", _u.kg), ("angle", _u.rad)]:
            if found := _find(reference):
                self._units[key] = found

        # Default to `stars` if no object unit is found.
        self._units["object"] = _find(stars) or stars

        # Infer velocity from length/time if not explicitly provided.
        velocity = _find(_u.km / _u.s)
        if velocity or (_find(_u.m) and _find(_u.s)):
            self._units["velocity"] = velocity or self._units["length"] / self._units["time"]

        # Infer density from object/length^3 if not explicitly provided.
        # Handles three cases:
        #   1. A pure inverse-volume unit (e.g. 1/pc^3) was given — combine with object unit.
        #   2. Object and length units were given — construct density from them.
        #   3. Only object unit changed — preserve the existing density's length unit.
        if density := _find(stars / _u.m**3):
            self._units["density"] = density
        elif inverse_volume := _find(1 / _u.m**3):
            self._units["density"] = self._units["object"] * inverse_volume
        elif _find(stars) and _find(_u.m):
            self._units["density"] = self._units["object"] / self._units["length"] ** 3
        elif _find(stars):
            density_length = next(u for u in self._units["density"].bases if u.is_equivalent(_u.m))
            self._units["density"] = self._units["object"] / density_length**3

        self._bases = list(self._units.values())


# Backwards compatibility aliases.
InvalidKeyException = TypeError
InvalidUnitException = TypeError
