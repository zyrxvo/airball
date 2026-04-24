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
"""Useful constants for `airball`."""

from . import units as _u

pi = _u.twopi / 2.0
twopi = _u.twopi
G = 1 * (_u.au**3 / _u.solMass / _u.yr2pi**2)
