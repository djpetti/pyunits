from typing import NamedTuple
import unittest.mock as mock

import pytest

from pyunits import mul_div_helpers
from pyunits.tests.testing_types import UnitFactory
from pyunits.types import CompoundTypeFactories


class Config(NamedTuple):
    """
    Encapsulates standard configuration for tests.
    :param compound_type_factories: The mocked CompoundTypeFactories.
    :param left_unit: The mocked left unit to use.
    :param right_unit: The mocked right unit use.
    :param mock_simplify: The mocked simply function to use.
    :param mock_wrap_numeric: The mocked WrapNumeric decorator.
    :param mock_unitless: The mocked Unitless constructor.
    """
    compound_type_factories: mock.Mock
    left_unit: mock.Mock
    right_unit: mock.Mock
    mock_simplify: mock.Mock
    mock_wrap_numeric: mock.Mock
    mock_unitless: mock.Mock


@pytest.fixture
def config(unit_factory: UnitFactory) -> Config:
    """
    Creates new configuration to use for testing.
    :param unit_factory: The factory to use for creating fake Units.
    :return: The new configuration that it created.
    """
    # Create the fake CompoundTypeFactories.
    mock_factories = mock.Mock(spec=CompoundTypeFactories)

    # Create the fake units.
    mock_left_unit = unit_factory("LeftUnit", 1.0)
    mock_right_unit = unit_factory("RightUnit", 2.0)

    with mock.patch(mul_div_helpers.__name__ + ".unit_analysis.simplify"
                    ) as mock_simplify, \
            mock.patch(mul_div_helpers.__name__ + ".WrapNumeric"
                       ) as mock_wrap_numeric, \
            mock.patch(mul_div_helpers.__name__ + ".Unitless"
                       ) as mock_unitless:
        # Make WrapNumeric into a transparent pass-through.
        mock_wrap_numeric.side_effect = lambda x: x

        yield Config(compound_type_factories=mock_factories,
                     left_unit=mock_left_unit,
                     right_unit=mock_right_unit,
                     mock_simplify=mock_simplify,
                     mock_wrap_numeric=mock_wrap_numeric,
                     mock_unitless=mock_unitless)

        # Finalization done upon exit from context manager.


@pytest.mark.parametrize("compatible", [True, False],
                         ids=["compatible", "not_compatible"])
def test_do_mul(config: Config, compatible: bool) -> None:
    """
    Tests that do_mul() works.
    :param config: The configuration to use for testing.
    :param compatible: Whether to make it look like the units are compatible.
    """
    # Arrange.
    # Set the unit compatibility.
    config.left_unit.type.is_compatible.return_value = compatible
    config.right_unit.type.is_compatible.return_value = compatible

    #  Act.
    product = mul_div_helpers.do_mul(config.compound_type_factories,
                                     config.left_unit, config.right_unit)

    # Assert.
    # It should have checked compatibility.
    config.right_unit.type.is_compatible.assert_called_once_with(
        config.left_unit.type)
    converted_right = config.right_unit
    if compatible:
        # It should have done the conversion.
        config.left_unit.type.assert_called_once_with(config.right_unit)
        converted_right = config.left_unit.type.return_value

    # It should have created the compound unit.
    config.compound_type_factories.mul.assert_called_once_with(
        config.left_unit.type, converted_right.type)
    mock_compound_type = config.compound_type_factories.mul.return_value
    mock_compound_type.apply_to.assert_called_once_with(
        config.left_unit, converted_right)

    # It should have attempted simplification.
    mul_unit = mock_compound_type.apply_to.return_value
    config.mock_simplify.assert_called_once_with(mul_unit,
                                                 config.compound_type_factories)
    assert product == config.mock_simplify.return_value


def test_do_div_compatible(config: Config) -> None:
    """
    Tests that to_div() works when the units to divide are compatible.
    :param config: The configuration to use for testing.
    """
    # Arrange.
    # Make it look like the units are compatible.
    config.left_unit.type.is_compatible.return_value = True
    config.right_unit.type.is_compatible.return_value = True

    # Make sure the right unit has a raw value for it to divide.
    converted_right = config.left_unit.type.return_value
    type(converted_right).raw = mock.PropertyMock(return_value=1.0)

    # Act.
    quotient = mul_div_helpers.do_div(config.compound_type_factories,
                                      config.left_unit, config.right_unit)

    # Assert.
    # It should have checked compatibility.
    config.right_unit.type.is_compatible.assert_called_once_with(
        config.left_unit.type)
    # It should have done the conversion.
    config.left_unit.type.assert_called_once_with(config.right_unit)

    # It should have created a new Unitless object.
    config.mock_unitless.assert_called_once_with(config.left_unit.raw /
                                                 converted_right.raw)

    # It should have returned that.
    assert quotient == config.mock_unitless.return_value


def test_do_div_incompatible(config: Config) -> None:
    """
    Tests that to_div() works when the units to divide are not compatible.
    :param config: The configuration to use for testing.
    """
    # Arrange.
    # Make it look like the units are incompatible.
    config.left_unit.type.is_compatible.return_value = False
    config.right_unit.type.is_compatible.return_value = False

    # Act.
    quotient = mul_div_helpers.do_div(config.compound_type_factories,
                                      config.left_unit, config.right_unit)

    # Assert.
    # It should have checked compatibility.
    config.right_unit.type.is_compatible.assert_called_once_with(
        config.left_unit.type)

    # It should have created the compound unit.
    config.compound_type_factories.div.assert_called_once_with(
        config.left_unit.type, config.right_unit.type)
    mock_compound_type = config.compound_type_factories.div.return_value
    mock_compound_type.apply_to.assert_called_once_with(
        config.left_unit, config.right_unit)

    # It should have attempted simplification.
    div_unit = mock_compound_type.apply_to.return_value
    config.mock_simplify.assert_called_once_with(div_unit,
                                                 config.compound_type_factories)
    assert quotient == config.mock_simplify.return_value
