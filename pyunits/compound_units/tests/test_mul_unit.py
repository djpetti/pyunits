from typing import NamedTuple
import unittest.mock as mock

import numpy as np

import pytest

from pyunits.compound_units import mul_unit
from pyunits.compound_units.compound_unit_type import CompoundUnitType
from pyunits.compound_units.operations import Operation
from pyunits.types import UnitValue
from pyunits.unit_interface import UnitInterface


class TestMulUnit:
    """
    Tests for the MulUnit class.
    """

    class UnitConfig(NamedTuple):
        """
        Represents configuration for the tests.
        :param mul_unit: The MulUnit under test.
        :param mock_unit_type: The mocked type of the MulUnit.
        :param mock_left_unit: The mocked left sub-unit.
        :param mock_right_unit: The mocked right sub-unit.
        """
        mul_unit: mul_unit.MulUnit
        mock_unit_type: mock.Mock
        mock_left_unit: mock.Mock
        mock_right_unit: mock.Mock

    @classmethod
    @pytest.fixture
    def config(cls) -> UnitConfig:
        """
        Creates new configuration for a test.
        :return: The configuration that it created,
        """
        # Create the fake unit type.
        mock_unit_type = mock.Mock(spec=CompoundUnitType)
        # Create the fake sub-units.
        mock_left_unit = mock.Mock(spec=UnitInterface)
        mock_right_unit = mock.Mock(spec=UnitInterface)

        # Create the fake unit.
        my_mul_unit = mul_unit.MulUnit(mock_unit_type, mock_left_unit,
                                       mock_right_unit)

        return cls.UnitConfig(mul_unit=my_mul_unit,
                              mock_unit_type=mock_unit_type,
                              mock_left_unit=mock_left_unit,
                              mock_right_unit=mock_right_unit)

    @pytest.mark.parametrize("mul_by", [10, 5.0, np.array([1, 2, 3]),
                                        [1, 2, 3]])
    def test_mul_numeric(self, config: UnitConfig,
                         mul_by: UnitValue) -> None:
        """
        Tests that the multiplication operation works correctly when a unit is
        multiplied by a numeric value.
        :param config: The configuration to use.
        :param mul_by: Value to multiply by.
        """
        # Arrange.
        # Set the raw value of the sub-units.
        mock_left_raw = mock.PropertyMock(return_value=2)
        mock_right_raw = mock.PropertyMock(return_value=3)
        type(config.mock_left_unit).raw = mock_left_raw
        type(config.mock_right_unit).raw = mock_right_raw

        expected_product = 2 * 3 * mul_by

        # Act.
        # Multiply by a numeric value.
        product = config.mul_unit * mul_by

        # Assert.
        # It should have created a new unit of the same type.
        assert config.mock_unit_type.call_count == 1
        # It should have been created with the raw product value.
        my_args, _ = config.mock_unit_type.call_args
        got_product = my_args[0]
        np.testing.assert_array_equal(expected_product, got_product)

        assert product == config.mock_unit_type.return_value

    def test_mul_compatible_unit(self, config: UnitConfig) -> None:
        """
        Tests that the multiplication operation works correctly when a unit is
        multiplied by another with a compatible type.
        :param config: The configuration to use.
        """
        # Arrange.
        # Mock the __class__ attribute of the fake CompoundUnitType.
        mock_unit_type_class = mock.Mock(spec=type)
        config.mock_unit_type.mock_add_spec(mock_unit_type_class)

        # Make another MulUnit to multiply by.
        other_unit = mock.Mock(spec=mul_unit.MulUnit)

        # Make sure converted units have the same type as their parent.
        converted = config.mock_unit_type.return_value
        mock_type_property = mock.PropertyMock(
            return_value=config.mock_unit_type)
        type(converted).type = mock_type_property

        # Act.
        product = config.mul_unit * other_unit

        # Assert.
        # It should have checked compatibility.
        other_unit.type.is_compatible.assert_called_once_with(
            config.mock_unit_type)

        # It should have converted the other unit.
        config.mock_unit_type.assert_called_once_with(other_unit)
        # It should have created a new compound unit.
        # functools.partial() will add an extra argument here.
        mock_unit_type_class.assert_called_once_with(
            Operation.MUL, config.mock_unit_type, config.mock_unit_type)

        compound_unit = mock_unit_type_class.return_value
        compound_unit.apply_to.assert_called_once_with(config.mul_unit,
                                                       converted)

        # It should have returned the compound unit.
        assert product == compound_unit.apply_to.return_value

    def test_mul_incompatible_unit(self, config: UnitConfig) -> None:
        """
        Tests that the multiplication operation works correctly when a unit is
        multiplied by another with an incompatible type.
        :param config: The configuration to use.
        """
        # Arrange.
        # Mock the __class__ attribute of the fake CompoundUnitType.
        mock_unit_type_class = mock.Mock(spec=type)
        config.mock_unit_type.mock_add_spec(mock_unit_type_class)

        # Make another Unit to multiply by.
        other_unit = mock.Mock(spec=UnitInterface)

        # Create a fake type for the other unit.
        other_type = mock.Mock()
        other_type_property = mock.PropertyMock(return_value=other_type)
        type(other_unit).type = other_type_property

        # Make it look like the types are incompatible.
        other_type.is_compatible.return_value = False

        # Act.
        product = config.mul_unit * other_unit

        # Assert.
        # It should have checked compatibility.
        other_type.is_compatible.assert_called_once_with(config.mock_unit_type)
        # It should not have attempted to convert.
        config.mock_unit_type.assert_not_called()

        # It should have created a new compound unit.
        mock_unit_type_class.assert_called_once_with(
            Operation.MUL, config.mock_unit_type, other_type)

        compound_unit = mock_unit_type_class.return_value
        compound_unit.apply_to.assert_called_once_with(config.mul_unit,
                                                       other_unit)

        # It should have returned the compound unit.
        assert product == compound_unit.apply_to.return_value

    def test_raw(self, config: UnitConfig) -> None:
        """
        Tests that we can successfully get the raw value.
        :param config: The configuration to use.
        """
        # Arrange.
        # Set reasonable raw values for the sub-units.
        mock_left_raw = mock.PropertyMock(return_value=6)
        mock_right_raw = mock.PropertyMock(return_value=7)
        type(config.mock_left_unit).raw = mock_left_raw
        type(config.mock_right_unit).raw = mock_right_raw

        # Act.
        product = config.mul_unit.raw

        # Assert.
        # It should have multiplied the raw values.
        assert product == 42

    def test_name(self, config: UnitConfig) -> None:
        """
        Tests that we can successfully get the name of the unit.
        :param config: The configuration to use.
        """
        # Arrange.
        # Set names for the sub-units.
        mock_left_name = mock.PropertyMock(return_value="N")
        mock_right_name = mock.PropertyMock(return_value="m")
        type(config.mock_left_unit).name = mock_left_name
        type(config.mock_right_unit).name = mock_right_name

        # Act.
        name = config.mul_unit.name

        # Assert.
        assert name == "Nm"
