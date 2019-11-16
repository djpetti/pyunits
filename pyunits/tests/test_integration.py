import math

import pytest

from examples import example_units as eu
from pyunits.compound_units import Mul, Div
from pyunits.unitless import Unitless

"""
Integration tests that use our suite of example units and test large amounts
of PyUnits functionality together.
"""

SquareMiles = Mul(eu.Miles, eu.Miles)
SquareMeters = Mul(eu.Meters, eu.Meters)
SquareInches = Mul(eu.Inches, eu.Inches)
CubicMeters = Mul(eu.Meters, SquareMeters)
CubicIn = Mul(eu.Inches, SquareInches)

Pascals = Div(eu.Joules, CubicMeters)
MilesPerIn3 = Div(eu.Miles, CubicIn)
InchesPerSecond = Div(eu.Inches, eu.Seconds)
MetersPerSecond = Div(eu.Meters, eu.Seconds)
MPerSSquared = Div(MetersPerSecond, eu.Seconds)


@pytest.mark.integration
def test_pi() -> None:
    """
    A test that performs the calculation outlined here:
    https://www.xkcd.com/687/
    """
    # Arrange.
    plank_energy = eu.Joules(1.956E9)
    core_pressure = Pascals(3.6E11)
    # Technically, this is the Prius C gas mileage. It seems that Priuses
    # (Priui?) have improved somewhat since the comic was published...
    # Note also that we don't really have the ability to do mpg directly yet,
    # so we do mi / in^3 and divide by 231. (1 gal = 231 in^3)
    prius_mileage = MilesPerIn3(46 / 231)
    channel_width = eu.Kilometers(33.8)

    # Act.
    pi = plank_energy / core_pressure * prius_mileage / channel_width

    # Assert.
    # This calculation isn't going to give us *exactly* pi, but by massaging the
    # numbers, we can get pretty close.
    assert pi.type.is_compatible(Unitless)
    assert pi.raw == pytest.approx(3.14, abs=0.01)


@pytest.mark.integration
def test_fermirotica() -> None:
    """
    A test that performs the calculation outlined here:
    https://xkcd.com/563/
    """
    # Arrange.
    population_density = 18600 / SquareMiles(1.0)
    sex_frequency = 80 / eu.Years(1.0)
    sex_duration = eu.Minutes(30)

    # Act.
    denom_value = math.pi * population_density * sex_frequency * sex_duration
    fraction = 2.0 / denom_value

    # Assert.
    # The units should check out.
    fraction = SquareMeters(fraction)

    # Square root has to be done outside of PyUnits, since we don't support that
    # (yet).
    sex_radius = math.sqrt(fraction.raw)
    # The correct value for these parameters is 139 m.
    assert sex_radius == pytest.approx(139.33, abs=0.01)


@pytest.mark.integration
def test_simple_conversions() -> None:
    """
    Tests that simple conversions between units work.
    """
    # Arrange.
    length1 = eu.Meters(42)
    length2 = eu.Inches(12)

    # Act.
    total_length = length1 + length2

    # Assert.
    assert total_length.raw == pytest.approx(42.3048)


@pytest.mark.integration
def test_division_unitless() -> None:
    """
    Tests that division works when we want to divide compatible units.
    """
    # Arrange.
    length1 = eu.Meters(10)
    length2 = eu.Centimeters(1000)

    # Act.
    quotient = length1 / length2

    # Assert.
    # It should be a unitless value.
    assert quotient.type.is_compatible(Unitless)
    # It should have divided correctly.
    assert quotient.raw == pytest.approx(1.0)


@pytest.mark.integration
def test_circle_area() -> None:
    """
    Tests that we can compute the area of a circle with units.
    """
    # Arrange.
    pi = Unitless(math.pi)
    radius = eu.Inches(5)

    # Act.
    area = pi * radius * radius

    # Assert.
    # The result should be in square inches.
    area = SquareInches(area)
    assert area.raw == pytest.approx(5 ** 2 * math.pi)


@pytest.mark.integration
def test_kinematics() -> None:
    """
    Tests that we can perform a basic kinematics calculation.
    """
    # Arrange.
    accel = MPerSSquared(-9.81)
    initial_vel = InchesPerSecond(50)
    initial_pos = eu.Centimeters(500)
    sample_time = eu.Seconds(2.0)

    # Act.
    sample_pos = 0.5 * accel * sample_time * sample_time + \
        initial_vel * sample_time + initial_pos

    # Assert.
    # We should be able to get the position in meters.
    sample_pos = eu.Meters(sample_pos)
    assert sample_pos.raw == pytest.approx(-12.08)


# TODO (Issue 8) Make this test succeed.
@pytest.mark.integration
@pytest.mark.xfail
def test_compound_compatible() -> None:
    """
    Tests that complicated compatibility checks between compound units succeed.
    """
    # Arrange.
    compound_type = Mul(Mul(eu.Meters, eu.Newtons),
                        Mul(eu.Kilograms, eu.Seconds))
    isomorphic_type = Mul(Mul(eu.Meters, eu.Kilograms),
                          Mul(eu.Seconds, eu.Newtons))

    # Act and assert.
    assert compound_type.is_compatible(isomorphic_type)
