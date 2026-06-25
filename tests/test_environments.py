import tempfile

import numpy as np
import pytest

import airball
import airball.units as u


@pytest.fixture
def custom_env():
    return airball.StellarEnvironment(
        stellar_density=10,
        velocity_dispersion=20,
        lower_mass_limit=0.08,
        upper_mass_limit=8,
        name="Test Environment",
    )


PRESETS = pytest.mark.parametrize(
    "env",
    [
        airball.OpenCluster(),
        airball.LocalNeighborhood(),
        airball.GlobularCluster(),
        airball.GalacticBulge(),
        airball.GalacticCore(),
        airball.StellarEnvironment(
            stellar_density=10,
            velocity_dispersion=20,
            lower_mass_limit=0.08,
            upper_mass_limit=8,
            name="Test Environment",
        ),
    ],
)


# ═══════════════════════════════════════════════════════════════════════════════
# region A) INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════


@PRESETS
def test_preset_environment_initialization(env):
    assert env is not None


def test_custom_environment_initialization(custom_env):
    env = custom_env
    assert env.density.value == pytest.approx(10)
    assert env.velocity_dispersion.value == pytest.approx(20)
    assert env.lower_mass_limit.value == pytest.approx(0.08)
    assert env.upper_mass_limit.value == pytest.approx(8)
    assert env.name == "Test Environment"


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(velocity_dispersion=20, lower_mass_limit=0.08, upper_mass_limit=8),
        dict(stellar_density=10, lower_mass_limit=0.08, upper_mass_limit=8),
        dict(stellar_density=10, velocity_dispersion=20, upper_mass_limit=8),
        dict(stellar_density=10, velocity_dispersion=20, lower_mass_limit=0.08),
    ],
)
def test_missing_required_parameters_raises(kwargs):
    with pytest.raises(AssertionError):
        airball.StellarEnvironment(**kwargs)


def test_initialization_with_units():
    env = airball.StellarEnvironment(
        stellar_density=10 * u.stars / u.pc**3,
        velocity_dispersion=20 * u.km / u.s,
        lower_mass_limit=0.08 * u.solMass,
        upper_mass_limit=8 * u.solMass,
    )
    assert env.density.unit.is_equivalent(u.stars / u.pc**3)
    assert env.velocity_dispersion.unit.is_equivalent(u.km / u.s)
    assert env.lower_mass_limit.unit.is_equivalent(u.solMass)
    assert env.upper_mass_limit.unit.is_equivalent(u.solMass)


# ═══════════════════════════════════════════════════════════════════════════════
# region B) PROPERTIES
# ═══════════════════════════════════════════════════════════════════════════════


@PRESETS
def test_mass_limits_ordered(env):
    assert env.lower_mass_limit < env.upper_mass_limit


@PRESETS
def test_mass_statistics_within_limits(env):
    assert env.lower_mass_limit <= env.median_mass <= env.upper_mass_limit
    assert env.lower_mass_limit <= env.mean_mass <= env.upper_mass_limit


@PRESETS
def test_positive_properties(env):
    assert env.maximum_impact_parameter.value > 0
    assert env.encounter_rate.value > 0
    assert env.velocity_mean.value > 0
    assert env.velocity_mode.value > 0


# ═══════════════════════════════════════════════════════════════════════════════
# region C) RANDOM STARS
# ═══════════════════════════════════════════════════════════════════════════════


@PRESETS
def test_random_star_returns_star(env):
    star = env.random_star()
    assert isinstance(star, airball.Star)


@PRESETS
def test_random_stars_returns_stars(env):
    stars = env.random_stars(size=5)
    assert isinstance(stars, airball.Stars)
    assert len(stars) == 5


@PRESETS
def test_random_stars_properties(env):
    stars = env.random_stars(size=100, seed=42)
    assert np.all(stars.m >= env.lower_mass_limit)
    assert np.all(stars.m <= env.upper_mass_limit)
    assert np.all(stars.v > 0 * u.km / u.s)
    assert np.all(stars.b >= 0 * u.au)
    assert np.all(stars.b <= env.maximum_impact_parameter)


def test_random_stars_reproducible_with_seed():
    env = airball.LocalNeighborhood(seed=99)
    stars1 = env.random_stars(size=10)
    stars2 = env.random_stars(size=10)
    assert np.all(stars1.m == stars2.m)
    assert np.all(stars1.b == stars2.b)
    assert np.all(stars1.v == stars2.v)


def test_random_stars_2d_shape():
    env = airball.OpenCluster()
    stars = env.random_stars(size=(3, 4))
    assert isinstance(stars, airball.Stars)
    assert stars.m.shape == (3, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# region D) ENCOUNTER TIMES
# ═══════════════════════════════════════════════════════════════════════════════


@PRESETS
@pytest.mark.parametrize("size,expected_shape", [(10, (10,)), ((3, 5), (3, 5))])
def test_encounter_times_shape(env, size, expected_shape):
    times = env.encounter_times(size)
    assert times.shape == expected_shape
    assert np.all(times > 0 * u.yr)


@PRESETS
def test_cumulative_encounter_times_is_nondecreasing(env):
    times = env.cumulative_encounter_times(20)
    assert np.all(np.diff(times) >= 0 * u.yr)


@PRESETS
def test_time_to_next_encounter_is_positive(env):
    t = env.time_to_next_encounter()
    assert t.value > 0


# ═══════════════════════════════════════════════════════════════════════════════
# region E) COPY AND EQUALITY
# ═══════════════════════════════════════════════════════════════════════════════


def test_copy_equals_original():
    env = airball.OpenCluster()
    env_copy = env.copy()
    assert env == env_copy
    assert env is not env_copy


def test_environments_with_different_densities_are_not_equal():
    env1 = airball.StellarEnvironment(stellar_density=10, velocity_dispersion=20, lower_mass_limit=0.08, upper_mass_limit=8)
    env2 = airball.StellarEnvironment(stellar_density=50, velocity_dispersion=20, lower_mass_limit=0.08, upper_mass_limit=8)
    assert env1 != env2


def test_save_and_load_roundtrip():
    env = airball.OpenCluster()
    with tempfile.NamedTemporaryFile(suffix=".se", delete=False) as f:
        filename = f.name
    env.save(filename)
    loaded = airball.StellarEnvironment(filename=filename)
    assert loaded == env


def test_save_invalid_filename_raises():
    env = airball.OpenCluster()
    with pytest.raises((ValueError, TypeError)):
        env.save(123)
