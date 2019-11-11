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
CubicMeters = Mul(eu.Meters, Mul(eu.Meters, eu.Meters))
CubicIn = Mul(eu.Inches, Mul(eu.Inches, eu.Inches))
Pascals = Div(eu.Joules, CubicMeters)
MilesPerIn3 = Div(eu.Miles, CubicIn)


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

