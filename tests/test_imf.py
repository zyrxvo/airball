"""
Tests for the airball.imf module.

Organized into two main sections:
  A) Functionality tests — validate the implementation and behavior of the module.
  B) Scientific accuracy tests — validate correctness against known analytic results.
"""

import numpy as np
import pytest

import airball.imf as imf
import airball.units as u
from airball.imf import IMF

# ═══════════════════════════════════════════════════════════════════════════════
# region A) FUNCTIONALITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


# ── IMF Initialization ────────────────────────────────────────────────────────


class TestIMFInit:
    """Tests for IMF class initialization."""

    def test_default_init(self):
        """Minimum required arguments produce correct defaults."""
        my_imf = IMF(min_mass=0.1, max_mass=100)
        assert my_imf.interpolating_points == int(1e5)
        assert my_imf.seed is None
        assert my_imf.unit == u.solMass
        assert isinstance(my_imf.initial_mass_function, imf.default_mass_function)

    def test_init_with_quantity_masses(self):
        """Accepts Quantity objects for min_mass and max_mass."""
        my_imf = IMF(min_mass=0.1 * u.solMass, max_mass=100 * u.solMass)
        assert my_imf.min_mass == 0.1 * u.solMass
        assert my_imf.max_mass == 100 * u.solMass

    def test_init_with_all_args(self):
        """All arguments are stored correctly."""
        mf = imf.salpeter_1955()
        my_imf = IMF(
            min_mass=0.5,
            max_mass=50,
            mass_function=mf,
            unit=u.solMass,
            interpolating_points=int(1e4),
            seed=42,
        )
        assert my_imf.min_mass.value == 0.5
        assert my_imf.max_mass.value == 50
        assert my_imf.interpolating_points == int(1e4)
        assert my_imf.seed == 42
        assert my_imf.unit == u.solMass

    def test_init_invalid_min_mass_zero(self):
        """min_mass <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="greater than 0"):
            IMF(min_mass=0, max_mass=100)

    def test_init_invalid_min_mass_negative(self):
        """Negative min_mass raises ValueError."""
        with pytest.raises(ValueError, match="greater than 0"):
            IMF(min_mass=-1, max_mass=100)

    def test_init_invalid_max_mass_zero(self):
        """max_mass <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="greater than 0"):
            IMF(min_mass=0.1, max_mass=0)

    def test_init_max_less_than_min(self):
        """max_mass < min_mass raises ValueError."""
        with pytest.raises(ValueError, match="greater than minimum"):
            IMF(min_mass=100, max_mass=0.1)

    def test_init_max_equal_min(self):
        """max_mass == min_mass raises ValueError."""
        with pytest.raises(ValueError, match="greater than minimum"):
            IMF(min_mass=1, max_mass=1)

    def test_unit_mismatch_raises(self):
        """mass_function.unit != IMF unit raises ValueError."""
        mf = imf.salpeter_1955()  # unit = solMass
        with pytest.raises(ValueError, match="does not match"):
            IMF(0.1, 100, mass_function=mf, unit=u.jupiterMass)

    def test_callable_without_unit_warns(self):
        """A plain callable without .unit emits a warning."""
        mf = lambda x: x**-2.35  # noqa: E731
        with pytest.warns(UserWarning, match="no 'unit' attribute"):
            IMF(0.1, 100, mass_function=mf)

    def test_non_callable_raises(self):
        """A non-callable mass_function raises an error."""
        with pytest.raises((ValueError, TypeError)):
            IMF(0.1, 100, mass_function="not a function")  # ty:ignore[invalid-argument-type]


# ── IMF Properties ────────────────────────────────────────────────────────────


class TestIMFProperties:
    """Tests for IMF properties and setters."""

    @pytest.fixture
    def default_imf(self) -> IMF:
        return IMF(0.1, 100, seed=42)

    def test_min_mass_setter(self, default_imf: IMF):
        """Setting min_mass recalculates and updates correctly."""
        default_imf.min_mass = 0.5
        assert default_imf.min_mass.value == 0.5

    def test_min_mass_setter_quantity(self, default_imf: IMF):
        """Setting min_mass with Quantity works."""
        default_imf.min_mass = 0.5 * u.solMass
        assert default_imf.min_mass.value == 0.5

    def test_min_mass_setter_invalid(self, default_imf: IMF):
        """Setting min_mass <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            default_imf.min_mass = 0

    def test_max_mass_setter(self, default_imf: IMF):
        """Setting max_mass recalculates and updates correctly."""
        default_imf.max_mass = 50
        assert default_imf.max_mass.value == 50

    def test_max_mass_setter_invalid(self, default_imf: IMF):
        """Setting max_mass <= min_mass raises ValueError."""
        with pytest.raises(ValueError):
            default_imf.max_mass = 0.05

    def test_seed_property(self, default_imf: IMF):
        """Seed can be set and read."""
        default_imf.seed = 123
        assert default_imf.seed == 123
        default_imf.seed = None
        assert default_imf.seed is None

    def test_interpolating_samples_setter(self, default_imf: IMF):
        """Setting interpolating_points recalculates."""
        default_imf.interpolating_points = 1000
        assert default_imf.interpolating_points == 1000

    def test_mass_range(self, default_imf: IMF):
        """mass_range returns [min_mass, max_mass]."""
        mr = default_imf.mass_range
        assert mr[0] == default_imf.min_mass
        assert mr[1] == default_imf.max_mass

    def test_median_mass_in_range(self, default_imf: IMF):
        """Median mass is within [min_mass, max_mass]."""
        med = default_imf.median_mass
        assert med >= default_imf.min_mass
        assert med <= default_imf.max_mass

    def test_mean_mass_in_range(self, default_imf: IMF):
        """Mean mass is within [min_mass, max_mass]."""
        mean = default_imf.mean_mass
        assert mean >= default_imf.min_mass
        assert mean <= default_imf.max_mass

    def test_mean_mass_in_range_below_one_solar_mass(self):
        """Mean mass is within [min_mass, max_mass] for a range entirely below 1 M_☉."""
        my_imf = IMF(0.01, 0.5, seed=42)
        mean = my_imf.mean_mass
        assert mean >= my_imf.min_mass
        assert mean <= my_imf.max_mass

    def test_mean_mass_salpeter_analytic(self):
        """Mean mass of Salpeter IMF matches the closed-form analytic result."""
        m_min, m_max, alpha = 0.1, 100.0, -2.35
        # E[m] = ∫m^(α+1)dm / ∫m^α dm = [(α+1)/(α+2)] * (m_max^(α+2) - m_min^(α+2)) / (m_max^(α+1) - m_min^(α+1))
        a1, a2 = alpha + 1, alpha + 2
        analytic_mean = (a1 / a2) * (m_max**a2 - m_min**a2) / (m_max**a1 - m_min**a1)
        my_imf = IMF(m_min, m_max, mass_function=imf.salpeter_1955())
        assert my_imf.mean_mass.value == pytest.approx(analytic_mean, rel=1e-12)


# ── IMF Methods ───────────────────────────────────────────────────────────────


class TestIMFMethods:
    """Tests for IMF methods."""

    @pytest.fixture
    def default_imf(self):
        return IMF(0.1, 100, seed=42)

    def test_random_mass_single(self, default_imf: IMF):
        """random_mass() returns a single Quantity."""
        m = default_imf.random_mass()
        assert isinstance(m, u.Quantity)
        assert m.unit == u.solMass

    def test_random_mass_array(self, default_imf: IMF):
        """random_mass(size=N) returns an array of N masses."""
        masses = default_imf.random_mass(size=100)
        assert len(masses) == 100
        assert all(m >= default_imf.min_mass for m in masses)
        assert all(m <= default_imf.max_mass for m in masses)

    def test_random_mass_tuple_shape(self, default_imf: IMF):
        """random_mass(size=(3,4)) returns correct shape."""
        masses = default_imf.random_mass(size=(3, 4))
        assert masses.shape == (3, 4)  # ty:ignore[unresolved-attribute]

    def test_random_mass_reproducible(self, default_imf: IMF):
        """Same seed produces same masses."""
        m1 = default_imf.random_mass(size=10, seed=99)
        m2 = default_imf.random_mass(size=10, seed=99)
        np.testing.assert_array_equal(m1.value, m2.value)  # ty:ignore[unresolved-attribute]

    def test_random_mass_different_seeds(self, default_imf: IMF):
        """Different seeds produce different masses."""
        m1 = default_imf.random_mass(size=10, seed=1)
        m2 = default_imf.random_mass(size=10, seed=2)
        assert not np.array_equal(m1.value, m2.value)  # ty:ignore[unresolved-attribute]

    def test_cdf_boundary_values(self, default_imf: IMF):
        """CDF is 0 at min_mass and 1 at max_mass."""
        assert default_imf.cdf(default_imf.min_mass.value) == 0.0
        assert default_imf.cdf(default_imf.max_mass.value) == 1.0

    def test_cdf_monotonic(self, default_imf: IMF):
        """CDF is monotonically non-decreasing."""
        x = np.geomspace(0.1, 100, 1000)
        cdf_vals = default_imf.cdf(x)
        assert np.all(np.diff(cdf_vals) >= 0)

    def test_cdf_outside_range(self, default_imf: IMF):
        """CDF returns 0 below min and 1 above max."""
        assert default_imf.cdf(0.01) == 0.0  # ty:ignore[invalid-argument-type]
        assert default_imf.cdf(1000) == 1.0  # ty:ignore[invalid-argument-type]

    def test_pdf_non_negative(self, default_imf: IMF):
        """PDF is non-negative within the range."""
        x = np.geomspace(0.1, 100, 1000)
        pdf_vals = default_imf.pdf(x)
        assert np.all(pdf_vals >= 0)

    def test_pdf_zero_outside_range(self, default_imf: IMF):
        """PDF is zero outside [min_mass, max_mass]."""
        assert default_imf.pdf(0.01) == 0.0  # ty:ignore[invalid-argument-type]
        assert default_imf.pdf(1000) == 0.0  # ty:ignore[invalid-argument-type]

    def test_masses_method(self, default_imf: IMF):
        """masses() returns correct number of geom-spaced values."""
        ms = default_imf.masses(100)
        assert len(ms) == 100
        assert ms[0] == 0.1
        assert ms[-1] == 100

    def test_masses_no_endpoint(self, default_imf: IMF):
        """masses(endpoint=False) excludes max_mass."""
        ms = default_imf.masses(100, endpoint=False)
        assert ms[-1] < 100

    def test_imf_method(self, default_imf: IMF):
        """imf() delegates to initial_mass_function."""
        x = np.array([1.0])
        np.testing.assert_array_equal(default_imf.imf(x), default_imf.initial_mass_function(x))

    def test_CDF_PDF_IMF_aliases(self, default_imf: IMF):
        """CDF, PDF, IMF properties alias the methods."""
        x = np.array([0.5, 1.0, 5.0])
        np.testing.assert_array_equal(default_imf.CDF(x), default_imf.cdf(x))
        np.testing.assert_array_equal(default_imf.PDF(x), default_imf.pdf(x))
        np.testing.assert_array_equal(default_imf.IMF(x), default_imf.imf(x))


# ── IMF Equality, Copy, and Repr ──────────────────────────────────────────────


class TestIMFEquality:
    """Tests for IMF __eq__, __hash__, copy, and repr."""

    def test_equality(self):
        """Identical IMFs are equal."""
        a = IMF(0.1, 100, seed=42)
        b = IMF(0.1, 100, seed=42)
        assert a == b

    def test_inequality_different_range(self):
        """IMFs with different ranges are not equal."""
        a = IMF(0.1, 100)
        b = IMF(0.2, 100)
        assert a != b

    def test_inequality_different_mf(self):
        """IMFs with different mass functions are not equal."""
        a = IMF(0.1, 100, mass_function=imf.salpeter_1955())
        b = IMF(0.1, 100, mass_function=imf.uniform())
        assert a != b

    def test_copy_equals_original(self):
        """A copy is equal to the original."""
        a = IMF(0.1, 100, seed=42)
        b = a.copy()
        assert a == b

    def test_copy_is_independent(self):
        """Modifying a copy does not affect the original."""
        a = IMF(0.1, 100, seed=42)
        b = a.copy()
        b.min_mass = 0.5
        assert a.min_mass.value == 0.1

    def test_repr_str(self):
        """repr/str returns a string (doesn't crash)."""
        my_imf = IMF(0.1, 100)
        s = repr(my_imf)
        assert isinstance(s, str)
        assert "IMF" in s

    def test_hash(self):
        """IMF objects can be hashed."""
        a = IMF(0.1, 100, seed=42)
        assert isinstance(hash(a), int)


# ── MassFunction Protocol ─────────────────────────────────────────────────────


class TestMassFunctionProtocol:
    """Tests for the MassFunction protocol."""

    def test_builtin_is_mass_function(self):
        """All builtin mass functions conform to the MassFunction protocol."""
        builtins = [
            imf.chabrier_2003_single(),
            imf.chabrier_2005_single(),
            imf.salpeter_1955(),
            imf.kroupa_1993(),
            imf.default_mass_function(),
            imf.uniform(),
            imf.power_law(alpha=-2.35),
            imf.broken_power_law(alpha=-1.3, beta=-2.35, m_0=0.5 * u.solMass),
            imf.lognormal(mu=-0.5, sigma=0.5),
            imf.loguniform(),
        ]
        for mf in builtins:
            assert isinstance(mf, imf.MassFunction), f"{mf.__class__.__name__} not MassFunction"

    def test_lambda_with_unit_is_mass_function(self):
        """A lambda with a .unit attribute conforms to the protocol."""
        mf = lambda x: x**-2  # noqa: E731
        mf.unit = u.solMass  # ty:ignore[unresolved-attribute]
        assert isinstance(mf, imf.MassFunction)

    def test_lambda_without_unit_is_not(self):
        """A lambda without .unit does not conform."""
        mf = lambda x: x**-2  # noqa: E731
        assert not isinstance(mf, imf.MassFunction)


# ── Mass Function Equality/Hash ───────────────────────────────────────────────


class TestMassFunctionEquality:
    """Tests for mass function __eq__ and __hash__."""

    def test_salpeter_equal(self):
        assert imf.salpeter_1955() == imf.salpeter_1955()

    def test_salpeter_not_equal(self):
        assert imf.salpeter_1955(xi_0=0.03) != imf.salpeter_1955(xi_0=0.05)

    def test_chabrier03_equal(self):
        assert imf.chabrier_2003_single() == imf.chabrier_2003_single()

    def test_kroupa_equal(self):
        assert imf.kroupa_1993() == imf.kroupa_1993()

    def test_power_law_equal(self):
        assert imf.power_law(alpha=-2.35) == imf.power_law(alpha=-2.35)

    def test_power_law_not_equal(self):
        assert imf.power_law(alpha=-2.35) != imf.power_law(alpha=-1.5)

    def test_uniform_equal(self):
        assert imf.uniform() == imf.uniform()

    def test_lognormal_equal(self):
        assert imf.lognormal(mu=-0.5, sigma=0.5) == imf.lognormal(mu=-0.5, sigma=0.5)

    def test_hashable(self):
        """All mass functions can be used in sets/dicts."""
        mfs = {
            imf.salpeter_1955(),
            imf.chabrier_2003_single(),
            imf.uniform(),
            imf.power_law(alpha=-2.0),
        }
        assert len(mfs) == 4


# ═══════════════════════════════════════════════════════════════════════════════
# region B) SCIENTIFIC ACCURACY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


# ── Chabrier (2003) ───────────────────────────────────────────────────────────


class TestChabrier2003:
    """Scientific validation of the Chabrier (2003) single-star IMF."""

    def test_normalization_at_0_7(self):
        """ξ(0.7 M_☉) ≈ 3.8×10⁻² M_☉⁻¹ pc⁻³ (paper Table 1, ±5%)."""
        c03 = imf.chabrier_2003_single()
        val = c03(0.7)
        assert val == pytest.approx(3.8e-2, rel=1e-2)

    def test_peak_near_characteristic_mass(self):
        """The linear-space PDF peaks near m_c (mode < m_c due to Jacobian)."""
        c03 = imf.chabrier_2003_single()
        x = np.geomspace(0.001, 1.0, 100000)
        y = c03(x)
        peak_mass = x[np.argmax(y)]
        # Peak of ξ(m) in linear space: m_peak = m_c * 10^(-σ²·ln10)
        # For m_c=0.079, σ=0.69: m_peak ≈ 0.0063 M_☉
        m_peak_analytic = 0.079 * 10 ** (-(0.69**2) * np.log(10))
        assert peak_mass == pytest.approx(m_peak_analytic, rel=1e-4)

    def test_monotone_decrease_above_0_3(self):
        """ξ(m) is monotonically decreasing for m > 0.3 M_☉."""
        c03 = imf.chabrier_2003_single()
        x = np.geomspace(0.3, 1.0, 1000)
        y = c03(x)
        assert np.all(np.diff(y) < 0)


# ── Chabrier (2005) ───────────────────────────────────────────────────────────


class TestChabrier2005:
    """Scientific validation of the Chabrier (2005) single-star IMF."""

    def test_same_functional_form_as_2003(self):
        """Identical functional form to 2003 but with different constants."""
        # When given the same constants, should produce the same result.
        c03 = imf.chabrier_2003_single()
        c05_as_03 = imf.chabrier_2005_single(A=0.158, m_c=0.079 * u.solMass, sigma=0.69)
        x = np.array([0.1, 0.3, 0.5, 0.8])
        np.testing.assert_allclose(c05_as_03(x), c03(x), rtol=1e-14)

    def test_higher_characteristic_mass(self):
        """m_c=0.2 M_☉ > 0.079 M_☉ shifts the peak to higher masses."""
        c03 = imf.chabrier_2003_single()
        c05 = imf.chabrier_2005_single()
        # The 2005 peak should be at a higher mass than the 2003 peak
        x = np.geomspace(0.001, 1.0, 100000)
        peak_03 = x[np.argmax(c03(x))]
        peak_05 = x[np.argmax(c05(x))]
        assert peak_05 > peak_03


# ── Salpeter (1955) ──────────────────────────────────────────────────────────


class TestSalpeter1955:
    """Scientific validation of the Salpeter (1955) IMF."""

    def test_power_law_slope(self):
        """d(log ξ)/d(log m) = -2.35 everywhere."""
        s = imf.salpeter_1955()
        m = np.array([0.5, 1.0, 5.0, 10.0, 50.0])
        log_xi = np.log10(s(m))
        log_m = np.log10(m)
        slopes = np.diff(log_xi) / np.diff(log_m)
        np.testing.assert_allclose(slopes, -2.35, rtol=1e-10)

    def test_value_at_1_solar_mass(self):
        """ξ(1 M_☉) = ξ₀ = 0.03 by definition."""
        s = imf.salpeter_1955()
        assert s(1.0) == 0.03

    def test_scaling(self):
        """ξ(m₂)/ξ(m₁) = (m₂/m₁)^(-2.35)."""
        s = imf.salpeter_1955()
        m1, m2 = 2.0, 8.0
        expected_ratio = (m2 / m1) ** -2.35
        actual_ratio = s(m2) / s(m1)
        assert actual_ratio == pytest.approx(expected_ratio, rel=1e-14)


# ── Kroupa, Tout & Gilmore (1993) ────────────────────────────────────────────


class TestKroupa1993:
    """Scientific validation of the Kroupa, Tout & Gilmore (1993) IMF."""

    def test_continuity_at_m2(self):
        """ξ is continuous at m₂ = 0.5 M_☉."""
        k = imf.kroupa_1993()
        eps = 1e-12
        left = k(0.5 - eps)
        right = k(0.5 + eps)
        assert left == pytest.approx(right, rel=1e-10)

    def test_continuity_at_m3(self):
        """ξ is continuous at m₃ = 1.0 M_☉."""
        k = imf.kroupa_1993()
        eps = 1e-12
        left = k(1.0 - eps)
        right = k(1.0 + eps)
        assert left == pytest.approx(right, rel=1e-10)

    def test_power_law_segments(self):
        """Each segment has the correct slope in log-log space."""
        k = imf.kroupa_1993()
        # Segment 1: m < 0.5
        m = np.array([0.15, 0.25, 0.4])
        slopes = np.diff(np.log10(k(m))) / np.diff(np.log10(m))
        np.testing.assert_allclose(slopes, -1.3, rtol=1e-6)
        # Segment 2: 0.5 < m < 1.0
        m = np.array([0.55, 0.7, 0.9])
        slopes = np.diff(np.log10(k(m))) / np.diff(np.log10(m))
        np.testing.assert_allclose(slopes, -2.2, rtol=1e-6)
        # Segment 3: m > 1.0
        m = np.array([1.5, 5.0, 20.0])
        slopes = np.diff(np.log10(k(m))) / np.diff(np.log10(m))
        np.testing.assert_allclose(slopes, -2.7, rtol=1e-6)

    def test_custom_parameters(self):
        """Custom α and m break values work correctly."""
        k = imf.kroupa_1993(
            alpha_1=1.0,
            alpha_2=2.0,
            alpha_3=3.0,
            m_2=0.3 * u.solMass,
            m_3=2.0 * u.solMass,
        )
        # Continuity at custom break points
        eps = 1e-12
        assert k(0.3 - eps) == pytest.approx(k(0.3 + eps), rel=1e-10)
        assert k(2.0 - eps) == pytest.approx(k(2.0 + eps), rel=1e-10)


# ── Default Mass Function ────────────────────────────────────────────────────


class TestDefaultMassFunction:
    """Scientific validation of the default piecewise Chabrier+Salpeter IMF."""

    def test_continuity_at_junction(self):
        """ξ is continuous at 1 M_☉ junction."""
        dmf = imf.default_mass_function()
        eps = 1e-12
        left = dmf(1.0 - eps)
        right = dmf(1.0 + eps)
        assert left == pytest.approx(right, rel=1e-10)

    def test_matches_chabrier_below_1(self):
        """Below 1 M_☉, matches Chabrier (2003)."""
        dmf = imf.default_mass_function()
        c03 = imf.chabrier_2003_single()
        x = np.array([0.1, 0.3, 0.5, 0.8, 0.99])
        np.testing.assert_allclose(dmf(x), c03(x), rtol=1e-10)

    def test_salpeter_slope_above_1(self):
        """Above 1 M_☉, slope is -2.35 in log-log space."""
        dmf = imf.default_mass_function()
        m = np.array([2.0, 5.0, 10.0, 50.0])
        log_xi = np.log10(dmf(m))
        log_m = np.log10(m)
        slopes = np.diff(log_xi) / np.diff(log_m)
        np.testing.assert_allclose(slopes, -2.35, rtol=1e-6)


# ── Generic Mass Functions ────────────────────────────────────────────────────


class TestGenericMassFunctions:
    """Scientific validation of generic/parametric mass functions."""

    def test_uniform_is_constant(self):
        """uniform() returns 1 for all masses."""
        uf = imf.uniform()
        x = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
        np.testing.assert_array_equal(uf(x), np.ones(5))

    def test_power_law_slope(self):
        """power_law(alpha) has correct log-log slope."""
        for alpha in [-3.0, -2.35, -1.0, 0.5]:
            pl = imf.power_law(alpha=alpha)
            m = np.array([0.5, 2.0, 10.0])
            slopes = np.diff(np.log10(pl(m))) / np.diff(np.log10(m))
            np.testing.assert_allclose(slopes, alpha, rtol=1e-10)

    def test_power_law_normalization(self):
        """power_law A parameter scales output linearly."""
        pl1 = imf.power_law(alpha=-2.0, A=1.0)
        pl2 = imf.power_law(alpha=-2.0, A=3.0)
        x = np.array([0.5, 1.0, 5.0])
        np.testing.assert_allclose(pl2(x) / pl1(x), 3.0, rtol=1e-14)

    def test_broken_power_law_continuity(self):
        """broken_power_law is continuous at m₀."""
        bpl = imf.broken_power_law(alpha=-1.3, beta=-2.35, m_0=0.5 * u.solMass)
        eps = 1e-10
        assert bpl(0.5 - eps) == pytest.approx(bpl(0.5 + eps), rel=1e-8)

    def test_broken_power_law_slopes(self):
        """broken_power_law has correct slopes in each segment."""
        bpl = imf.broken_power_law(alpha=-1.3, beta=-2.35, m_0=0.5 * u.solMass)
        # Below m_0
        m_lo = np.array([0.1, 0.2, 0.4])
        slopes_lo = np.diff(np.log10(bpl(m_lo))) / np.diff(np.log10(m_lo))
        np.testing.assert_allclose(slopes_lo, -1.3, rtol=1e-6)
        # Above m_0
        m_hi = np.array([1.0, 5.0, 20.0])
        slopes_hi = np.diff(np.log10(bpl(m_hi))) / np.diff(np.log10(m_hi))
        np.testing.assert_allclose(slopes_hi, -2.35, rtol=1e-6)

    def test_lognormal_matches_chabrier(self):
        """lognormal with Chabrier params is identical to chabrier_2003_single."""
        c03 = imf.chabrier_2003_single()
        ln_mf = imf.lognormal(mu=np.log10(0.079), sigma=0.69, A=0.158)
        x = np.geomspace(0.05, 1.0, 100)
        np.testing.assert_allclose(ln_mf(x), c03(x), rtol=1e-14)

    def test_lognormal_symmetry_in_log_space(self):
        """lognormal is symmetric around μ in log₁₀(m) space."""
        mu = np.log10(0.3)
        ln_mf = imf.lognormal(mu=mu, sigma=0.5)
        # ξ(log m) = ξ(m) * m * ln(10) should be symmetric
        m_left = 10 ** (mu - 0.2)
        m_right = 10 ** (mu + 0.2)
        xi_log_left = ln_mf(m_left) * m_left * np.log(10)
        xi_log_right = ln_mf(m_right) * m_right * np.log(10)
        assert xi_log_left == pytest.approx(xi_log_right, rel=1e-14)

    def test_loguniform_slope(self):
        """loguniform has slope -1 in log-log space (ξ ∝ 1/m)."""
        lu = imf.loguniform()
        m = np.array([0.1, 1.0, 10.0, 100.0])
        slopes = np.diff(np.log10(lu(m))) / np.diff(np.log10(m))
        np.testing.assert_allclose(slopes, -1.0, rtol=1e-14)

    def test_loguniform_equal_probability_per_decade(self):
        """loguniform gives equal probability per decade when integrated."""
        my_imf = IMF(0.1, 100, mass_function=imf.loguniform())
        # CDF(1) - CDF(0.1) should equal CDF(10) - CDF(1)
        decade1 = my_imf.cdf(1.0) - my_imf.cdf(0.1)  # ty:ignore[invalid-argument-type]
        decade2 = my_imf.cdf(10.0) - my_imf.cdf(1.0)  # ty:ignore[invalid-argument-type]
        decade3 = my_imf.cdf(100.0) - my_imf.cdf(10.0)  # ty:ignore[invalid-argument-type]
        assert decade1 == pytest.approx(decade2, rel=1e-11)
        assert decade2 == pytest.approx(decade3, rel=1e-11)


# ── PDF/CDF Integration Tests ────────────────────────────────────────────────


class TestIntegration:
    """Tests that PDF integrates to 1 and CDF spans [0, 1]."""

    @pytest.mark.parametrize(
        "mass_function",
        [
            imf.salpeter_1955(),
            imf.chabrier_2003_single(),
            imf.chabrier_2005_single(),
            imf.kroupa_1993(),
            imf.default_mass_function(),
            imf.uniform(),
            imf.power_law(alpha=-2.0),
            imf.lognormal(mu=-0.5, sigma=0.5),
            imf.loguniform(),
        ],
    )
    def test_pdf_integrates_to_one(self, mass_function):
        """∫ pdf(m) dm ≈ 1 over [min_mass, max_mass] for all mass functions."""
        my_imf = IMF(0.1, 100, mass_function=mass_function)
        # Numerical integration using trapezoidal rule on a fine grid
        x = np.geomspace(0.1, 100, 50000)
        pdf_vals = my_imf.pdf(x)
        integral = np.trapezoid(pdf_vals, x)
        assert integral == pytest.approx(1.0, rel=1e-7)

    @pytest.mark.parametrize(
        "mass_function",
        [
            imf.salpeter_1955(),
            imf.default_mass_function(),
            imf.uniform(),
        ],
    )
    def test_cdf_spans_zero_to_one(self, mass_function):
        """CDF goes from 0 to 1 across the mass range."""
        my_imf = IMF(0.1, 100, mass_function=mass_function)
        assert my_imf.cdf(0.1) == 0.0  # ty:ignore[invalid-argument-type]
        assert my_imf.cdf(100.0) == 1.0  # ty:ignore[invalid-argument-type]


# ── Sampling Distribution Tests ──────────────────────────────────────────────


class TestSamplingDistribution:
    """Tests that random_mass sampling produces correct distributions."""

    def test_uniform_distribution_flat(self):
        """Uniform IMF produces roughly uniform samples."""
        my_imf = IMF(1, 10, mass_function=imf.uniform(), seed=0)
        samples = my_imf.random_mass(size=50000).value  # ty:ignore[unresolved-attribute]
        # Split into equal bins and check roughly equal counts
        counts, _ = np.histogram(samples, bins=9, range=(1, 10))
        expected = 50000 / 9
        # No bin deviates by more than 3%
        assert np.all(np.abs(counts - expected) / expected < 0.03)

    def test_salpeter_median(self):
        """Salpeter IMF [0.1, 100] has median close to analytic value."""
        # For ξ(m) ∝ m^α, CDF(m) ∝ m^(α+1), so median satisfies
        # (m_med^(α+1) - m_min^(α+1)) / (m_max^(α+1) - m_min^(α+1)) = 0.5
        alpha = -2.35
        a = alpha + 1  # = -1.35
        m_min, m_max = 0.1, 100.0
        m_med_analytic = (0.5 * (m_max**a - m_min**a) + m_min**a) ** (1 / a)
        my_imf = IMF(m_min, m_max, mass_function=imf.salpeter_1955())
        assert my_imf.median_mass.value == pytest.approx(m_med_analytic, rel=1e-12)

    def test_default_imf_median_reasonable(self):
        """Default IMF [0.1, 100] median ≈ 0.2–0.4 M_☉ (literature value)."""
        my_imf = IMF(0.1, 100)
        med = my_imf.median_mass.value
        assert 0.1 < med < 0.5


# ── Quantity/Unit Handling ────────────────────────────────────────────────────


class TestUnitHandling:
    """Tests that mass functions handle Quantity inputs correctly."""

    def test_quantity_input_same_as_float(self):
        """Passing Quantity gives same result as passing float."""
        mf = imf.salpeter_1955()
        val_float = mf(2.0)
        val_qty = mf(2.0 * u.solMass)
        assert val_float == pytest.approx(val_qty, rel=1e-14)

    def test_array_input(self):
        """Mass functions handle numpy arrays."""
        mf = imf.chabrier_2003_single()
        x = np.array([0.1, 0.5, 1.0])
        result = mf(x)
        assert result.shape == (3,)  # ty:ignore[unresolved-attribute]
        assert np.all(result > 0)

    def test_scalar_input(self):
        """Mass functions handle scalar floats."""
        mf = imf.chabrier_2003_single()
        result = mf(0.5)
        assert np.isscalar(result) or result.shape == ()  # ty:ignore[unresolved-attribute]

    def test_all_builtins_have_solmass_unit(self):
        """All builtin mass functions declare unit = solMass."""
        builtins = [
            imf.chabrier_2003_single(),
            imf.chabrier_2005_single(),
            imf.salpeter_1955(),
            imf.kroupa_1993(),
            imf.default_mass_function(),
            imf.uniform(),
            imf.power_law(alpha=-2.0),
            imf.broken_power_law(alpha=-1.3, beta=-2.35, m_0=0.5 * u.solMass),
            imf.lognormal(mu=-0.5, sigma=0.5),
            imf.loguniform(),
        ]
        for mf in builtins:
            assert mf.unit == u.solMass, f"{mf.__class__.__name__}.unit != solMass"
