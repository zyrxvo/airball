"""Tests for the airball.units module."""

import pytest

import airball.units as u
from airball.units import UnitSet, is_unit, isUnit

# ═══════════════════════════════════════════════════════════════════════════════
# region A) MODULE-LEVEL HELPERS AND CUSTOM UNITS
# ═══════════════════════════════════════════════════════════════════════════════


class TestIsUnit:
    """Tests for the is_unit helper and its alias."""

    @pytest.mark.parametrize("unit", [u.au, u.yr, u.solMass, u.rad, u.km / u.s, u.stars])
    def test_units_return_true(self, unit):
        """Astropy Unit objects are recognized as units."""
        assert is_unit(unit) is True

    @pytest.mark.parametrize("non_unit", [1, 2.5, "au", None, [u.au]])
    def test_non_units_return_false(self, non_unit):
        """Non-unit objects are not recognized as units."""
        assert is_unit(non_unit) is False

    def test_alias(self):
        """Verify isUnit is a backwards-compatibility alias for is_unit."""
        assert isUnit is is_unit


class TestCustomUnits:
    """Tests for custom unit definitions."""

    def test_yr2pi_equivalent_to_time(self):
        """yr2pi is a time unit."""
        assert u.yr2pi.is_equivalent(u.s)

    def test_yrtwopi_equivalent_to_yr2pi(self):
        """Yrtwopi and yr2pi are equivalent."""
        assert u.yrtwopi.is_equivalent(u.yr2pi)

    def test_stars_not_equivalent_to_length(self):
        """Stars is not a length unit."""
        assert not u.stars.is_equivalent(u.m)

    def test_msun_alias(self):
        """The 'msun' alias resolves to solMass."""
        assert (1 * u.Unit("msun")).to(u.solMass).value == 1


# ═══════════════════════════════════════════════════════════════════════════════
# region B) UNITSET INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestUnitSetInit:
    """Tests for UnitSet initialization."""

    def test_default_init(self):
        """Default UnitSet has expected units for all keys."""
        us = UnitSet()
        assert us.length == u.au
        assert us.time == u.Myr
        assert us.mass == u.solMass
        assert us.angle == u.rad
        assert us.velocity == u.km / u.s
        assert us.object == u.stars
        assert us.density.to_string() == (u.stars / u.pc**3).to_string()

    def test_init_with_empty_list(self):
        """Empty list preserves defaults."""
        us = UnitSet([])
        assert us.length == u.au

    def test_init_with_none(self):
        """None preserves defaults."""
        us = UnitSet(None)
        assert us.length == u.au

    def test_init_with_units(self):
        """Providing units overrides the corresponding defaults."""
        us = UnitSet([u.pc, u.Myr])
        assert us.length == u.pc
        assert us.time == u.Myr

    def test_init_from_another_unitset(self):
        """Copying from another UnitSet produces an equal UnitSet."""
        us1 = UnitSet([u.pc, u.Myr])
        us2 = UnitSet(us1)
        assert us1 == us2
        assert us1 is not us2

    @pytest.mark.parametrize("bad", ["au", 123, {"a": "dict"}, {"a", "set"}])
    def test_init_invalid_type_raises(self, bad):
        """Non-list, non-UnitSet input raises TypeError."""
        with pytest.raises(TypeError, match="unit_system must be a list"):
            UnitSet(bad)

    def test_init_list_with_non_unit_raises(self):
        """A list containing non-Unit items raises TypeError."""
        with pytest.raises(TypeError, match="All items in unit_system must be Astropy Units"):
            UnitSet([u.au, "str"])  # ty: ignore[invalid-argument-type]


# ═══════════════════════════════════════════════════════════════════════════════
# region C) UNIT INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════


class TestUnitInference:
    """Tests for automatic inference of dependent units."""

    def test_velocity_inferred_from_length_and_time(self):
        """Velocity is length/time when both are given."""
        us = UnitSet([u.pc, u.Myr])
        assert us.velocity.is_equivalent(u.km / u.s)
        assert us.velocity == u.pc / u.Myr

    def test_velocity_explicit_overrides_inference(self):
        """An explicit velocity unit takes priority over inference."""
        us = UnitSet([u.pc, u.Myr, u.m / u.s])
        assert us.velocity == u.m / u.s

    def test_density_inferred_from_object_and_length(self):
        """Density is object/length^3 when both are given."""
        us = UnitSet([u.stars, u.au])
        assert us.density.is_equivalent(u.stars / u.m**3)

    def test_density_from_inverse_volume(self):
        """An inverse-volume unit is combined with the object unit."""
        us = UnitSet([u.pc**-3])
        assert us.density.is_equivalent(u.stars / u.m**3)

    def test_density_explicit(self):
        """An explicit density unit is used as-is."""
        us = UnitSet([u.stars / u.au**3])
        assert us.density.to_string() == (u.stars / u.au**3).to_string()

    def test_object_defaults_to_stars(self):
        """Object defaults to stars when not provided."""
        us = UnitSet([u.pc])
        assert us.object == u.stars

    def test_setting_wit_non_unit_raises(self):
        """A non-Unit raises TypeError."""
        us = UnitSet()
        with pytest.raises(TypeError, match="All items in unit_system must be Astropy Units"):
            us.length = "meters"  # ty: ignore[invalid-assignment]


# ═══════════════════════════════════════════════════════════════════════════════
# region D) DICT-LIKE ACCESS
# ═══════════════════════════════════════════════════════════════════════════════


class TestUnitSetAccess:
    """Tests for dict-like access on UnitSet."""

    @pytest.fixture
    def us(self):
        """Create a default UnitSet."""
        return UnitSet()

    def test_getitem(self, us):
        """Bracket access retrieves the correct unit."""
        assert us["length"] == u.au

    def test_getitem_invalid_key_raises(self, us):
        """Missing key raises KeyError."""
        with pytest.raises(KeyError):
            us["nonexistent"]

    def test_getitem_non_string_raises(self, us):
        """Non-string key raises TypeError."""
        with pytest.raises(TypeError, match="Key must be a string"):
            us[0]

    def test_setitem(self, us):
        """Bracket assignment updates a known unit."""
        us["length"] = u.pc
        assert us["length"] == u.pc

    def test_setitem_unknown_key_raises(self, us):
        """Unknown key raises KeyError on assignment."""
        with pytest.raises(KeyError, match="Unknown unit key"):
            us["custom"] = u.pc

    def test_setitem_non_string_key_raises(self, us):
        """Non-string key raises TypeError on assignment."""
        with pytest.raises(TypeError, match="Key must be a string"):
            us[0] = u.pc

    def test_setitem_non_unit_value_raises(self, us):
        """Non-unit value raises TypeError on assignment."""
        with pytest.raises(TypeError, match="valid Astropy Unit"):
            us["length"] = 42

    def test_keys(self, us):
        """keys() returns all expected unit names."""
        assert "length" in us
        assert "velocity" in us

    def test_values(self, us):
        """values() contains the default length unit."""
        assert u.au in us.values()

    def test_items(self, us):
        """items() yields correct key-value pairs."""
        items = dict(us.items())
        assert items["mass"] == u.solMass

    def test_iter_yields_keys(self, us):
        """Iterating yields unit keys."""
        keys = list(us)
        assert "length" in keys
        assert "mass" in keys


# ═══════════════════════════════════════════════════════════════════════════════
# region E) PROPERTY SETTERS
# ═══════════════════════════════════════════════════════════════════════════════


class TestUnitSetSetters:
    """Tests for property setters triggering unit_system recalculation."""

    def test_set_length(self):
        """Setting length updates the unit."""
        us = UnitSet()
        us.length = u.pc
        assert us.length == u.pc

    def test_set_time(self):
        """Setting time updates the unit."""
        us = UnitSet()
        us.time = u.Myr
        assert us.time == u.Myr

    def test_set_mass(self):
        """Setting mass updates the unit."""
        us = UnitSet()
        us.mass = u.jupiterMass
        assert us.mass == u.jupiterMass

    def test_set_angle(self):
        """Setting angle updates the unit."""
        us = UnitSet()
        us.angle = u.deg
        assert us.angle == u.deg

    def test_set_velocity(self):
        """Setting velocity updates the unit."""
        us = UnitSet()
        us.velocity = u.m / u.s
        assert us.velocity == u.m / u.s

    def test_set_density(self):
        """Setting density updates the unit."""
        us = UnitSet()
        us.density = u.stars / u.au**3
        assert us.density.to_string() == (u.stars / u.au**3).to_string()

    def test_set_object(self):
        """Setting object to stars preserves it."""
        us = UnitSet()
        us.object = u.stars
        assert us.object == u.stars

    def test_set_object_non_equivalent_defaults_to_stars(self):
        """A non-equivalent object unit falls back to stars."""
        custom = u.def_unit("planets")
        us = UnitSet()
        us.object = custom
        assert us.object == u.stars


# ═══════════════════════════════════════════════════════════════════════════════
# region F) EQUALITY, HASHING, AND STRING REPRESENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════


class TestUnitSetComparisons:
    """Tests for equality, hashing, and string output."""

    def test_equal_unitsets(self):
        """Two default UnitSets are equal."""
        assert UnitSet() == UnitSet()

    def test_unequal_unitsets(self):
        """UnitSets with different units are not equal."""
        assert UnitSet([u.pc]) != UnitSet()

    def test_not_equal_to_non_unitset(self):
        """A UnitSet is not equal to a non-UnitSet."""
        assert UnitSet() != "not a UnitSet"

    def test_hash_equal_for_equal_unitsets(self):
        """Equal UnitSets have equal hashes."""
        assert hash(UnitSet()) == hash(UnitSet())

    def test_hash_differs_for_different_unitsets(self):
        """Different UnitSets have different hashes."""
        assert hash(UnitSet([u.pc])) != hash(UnitSet())

    def test_usable_as_dict_key(self):
        """UnitSets can be used as dictionary keys."""
        us = UnitSet()
        d = {us: "value"}
        assert d[UnitSet()] == "value"

    def test_str(self):
        """str() includes unit key names."""
        s = str(UnitSet())
        assert "length" in s
        assert "mass" in s

    def test_repr(self):
        """repr() includes unit key names."""
        r = repr(UnitSet())
        assert "length" in r
        assert "mass" in r


# ═══════════════════════════════════════════════════════════════════════════════
# region G) FIND, DECOMPOSE, AND BASES
# ═══════════════════════════════════════════════════════════════════════════════


class TestFind:
    """Tests for the find method."""

    def test_find_known_unit(self):
        """Returns the category key for a known unit."""
        us = UnitSet()
        assert us.find(u.au) == "length"
        assert us.find(u.solMass) == "mass"
        assert us.find(u.stars) == "object"

    def test_find_equivalent_unit(self):
        """Returns the category key for an equivalent unit."""
        us = UnitSet()
        assert us.find(u.pc) == "length"
        assert us.find(u.s) == "time"

    def test_find_unknown_unit(self):
        """Returns None for a unit not in any category."""
        us = UnitSet()
        assert us.find(u.ampere) is None


class TestDecompose:
    """Tests for decompose and bases."""

    def test_bases_matches_unit_values(self):
        """Bases list matches the unit values."""
        us = UnitSet()
        assert set(us.bases) == set(us.values())

    def test_decompose_quantity(self):
        """decompose() reduces a quantity to base units."""
        us = UnitSet()
        q = 1.0 * u.km
        result = us.decompose(q)
        assert result.unit.is_equivalent(u.m)


# ═══════════════════════════════════════════════════════════════════════════════
# region H) BACKWARDS COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════════


class TestBackwardsCompatibility:
    """Tests for backwards compatibility aliases."""

    def test_arbitrary_attribute_raises(self):
        """Arbitrary attribute assignment is prevented by __slots__."""
        us = UnitSet()
        with pytest.raises(AttributeError):
            us.custom = 5  # ty: ignore[unresolved-attribute]

    def test_invalid_key_exception_alias(self):
        """InvalidKeyException is an alias for TypeError."""
        assert u.InvalidKeyException is TypeError

    def test_invalid_unit_exception_alias(self):
        """InvalidUnitException is an alias for TypeError."""
        assert u.InvalidUnitException is TypeError
