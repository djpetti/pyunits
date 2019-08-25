from typing import NamedTuple
import unittest.mock as mock

import pytest

from pyunits.compound_units import mul_unit
from pyunits.compound_units.compound_unit_type import CompoundUnitType
from pyunits.compound_units.operations import Operation
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
