from typing import NamedTuple
import unittest.mock as mock

import pytest

from pyunits import arithmetic_helpers
from pyunits.tests.testing_types import UnitFactory
from pyunits.types import CompoundTypeFactories
from pyunits.unitless import Unitless


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

    with mock.patch(arithmetic_helpers.__name__ + ".unit_analysis.simplify"
                    ) as mock_simplify, \
            mock.patch(arithmetic_helpers.__name__ + ".WrapNumeric"
                       ) as mock_wrap_numeric, \
            mock.patch(arithmetic_helpers.__name__ + ".Unitless"
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
    product = arithmetic_helpers.do_mul(config.compound_type_factories,
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
    quotient = arithmetic_helpers.do_div(config.compound_type_factories,
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
    quotient = arithmetic_helpers.do_div(config.compound_type_factories,
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


@pytest.mark.parametrize("left_unitless", [False, True],
                         ids=["left_not_unitless", "left_unitless"])
@pytest.mark.parametrize("right_unitless", [False, True],
                         ids=["right_not_unitless", "right_unitless"])
def test_do_add(config: Config, left_unitless: bool, right_unitless: bool
                ) -> None:
    """
    Tests that do_add() works.
    :param config: The configuration to use for testing.
    :param left_unitless: Whether the left value should be unitless.
    :param right_unitless: Whether the right value should be unitless.
    """
    # Arrange.
    # Correctly set which parameters are unitless.
    config.left_unit.type.is_compatible.return_value = left_unitless
    config.right_unit.type.is_compatible.return_value = right_unitless

    # Set the correct raw value for the converted units.
    left_not_unitless = config.left_unit
    right_not_unitless = config.right_unit
    if left_unitless:
        left_not_unitless = config.right_unit.type.return_value
        type(left_not_unitless).raw = mock.PropertyMock(
            return_value=config.left_unit.raw)

    # It will convert a maximum of one of them, hence the logic here.
    elif right_unitless:
        right_not_unitless = config.left_unit.type.return_value
        type(right_not_unitless).raw = mock.PropertyMock(
            return_value=config.right_unit.raw)

    # We will do an additional conversion to force both operands to the same
    # units. Correctly set the raw value for this.
    converted_right = left_not_unitless.type.return_value
    type(converted_right).raw = mock.PropertyMock(
        return_value=left_not_unitless.raw)

    # Act.
    unit_sum = arithmetic_helpers.do_add(config.left_unit, config.right_unit)

    # Assert.
    if left_unitless:
        # It should have converted the left unit.
        config.right_unit.type.assert_any_call(config.left_unit.raw)
    elif right_unitless:
        # It should have converted the right unit.
        config.left_unit.type.assert_any_call(config.right_unit.raw)

    # It should have done the addition.
    left_not_unitless.type.assert_any_call(right_not_unitless)
    right_not_unitless = left_not_unitless.type.return_value
    left_not_unitless.type.assert_any_call(left_not_unitless.raw +
                                           right_not_unitless.raw)
    assert unit_sum == left_not_unitless.type.return_value
