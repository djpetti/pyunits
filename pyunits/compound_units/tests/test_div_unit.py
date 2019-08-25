from typing import NamedTuple
import unittest.mock as mock

import numpy as np

import pytest

from pyunits.compound_units import div_unit
from pyunits.compound_units.compound_unit_type import CompoundUnitType
from pyunits.compound_units.operations import Operation
from pyunits.types import UnitValue
from pyunits.unit_interface import UnitInterface


class TestDivUnit:
    """
    Tests for the DivUnit class.
    """

    class UnitConfig(NamedTuple):
        """
        Represents configuration for the tests.
        :param div_unit: The DivUnit under test.
        :param mock_unit_type: The mocked type of the MulUnit.
        :param mock_left_unit: The mocked left sub-unit.
        :param mock_right_unit: The mocked right sub-unit.
        """
        div_unit: div_unit.DivUnit
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
        my_div_unit = div_unit.DivUnit(mock_unit_type, mock_left_unit,
                                       mock_right_unit)

        return cls.UnitConfig(div_unit=my_div_unit,
                              mock_unit_type=mock_unit_type,
                              mock_left_unit=mock_left_unit,
                              mock_right_unit=mock_right_unit)

    @pytest.mark.parametrize("div_by", [10, 5.0, np.array([1, 2, 3])])
    def test_div_numeric(self, config: UnitConfig,
                         div_by: UnitValue) -> None:
        """
        Tests that the division operation works correctly when a unit is
        divided by a numeric value.
        :param config: The configuration to use.
        :param div_by: Value to divide by.
        """
        # Arrange.
        left_sub_value = div_by * 20
        right_sub_value = div_by * 3

        # Set the raw value of the sub-units.
        mock_left_raw = mock.PropertyMock(return_value=left_sub_value)
        mock_right_raw = mock.PropertyMock(return_value=right_sub_value)
        type(config.mock_left_unit).raw = mock_left_raw
        type(config.mock_right_unit).raw = mock_right_raw

        expected_quotient = left_sub_value / right_sub_value / div_by

        # Act.
        # Divide by a numeric value.
        quotient = config.div_unit / div_by

        # Assert.
        # It should have created a new unit of the same type.
        assert config.mock_unit_type.call_count == 1
        # It should have been created with the raw quotient value.
        my_args, _ = config.mock_unit_type.call_args
        got_quotient = my_args[0]
        np.testing.assert_array_almost_equal(expected_quotient, got_quotient)

        assert quotient == config.mock_unit_type.return_value

    def test_div_compatible_unit(self, config: UnitConfig) -> None:
        """
        Tests that the multiplication operation works correctly when a unit is
        divided by another with a compatible type.
        :param config: The configuration to use.
        """
        # Arrange.
        # Mock the __class__ attribute of the fake CompoundUnitType.
        mock_unit_type_class = mock.Mock(spec=type)
        config.mock_unit_type.mock_add_spec(mock_unit_type_class)

        # Set the raw value of the sub-units.
        mock_left_raw = mock.PropertyMock(return_value=100)
        mock_right_raw = mock.PropertyMock(return_value=5)
        type(config.mock_left_unit).raw = mock_left_raw
        type(config.mock_right_unit).raw = mock_right_raw

        # Make another DivUnit to divide by.
        other_unit = mock.Mock(spec=div_unit.DivUnit)

        # Make sure converted units have the same type as their parent.
        converted = config.mock_unit_type.return_value
        mock_type_property = mock.PropertyMock(
            return_value=config.mock_unit_type)
        type(converted).type = mock_type_property
        # Mock the raw property too.
        mock_converted_raw = mock.PropertyMock(return_value=4)
        type(converted).raw = mock_converted_raw

        expected_quotient = 100 / 5 / 4

        # Act.
        quotient = config.div_unit / other_unit

        # Assert.
        # It should have checked compatibility.
        other_unit.type.is_compatible.assert_called_once_with(
            config.mock_unit_type)

        # It should have converted the other unit.
        config.mock_unit_type.assert_called_once_with(other_unit)

        # It should have returned a unitless value.
        assert quotient == pytest.approx(expected_quotient)

    def test_div_incompatible_unit(self, config: UnitConfig) -> None:
        """
        Tests that the division operation works correctly when a unit is
        divided by another with an incompatible type.
        :param config: The configuration to use.
        """
        # Arrange.
        # Mock the __class__ attribute of the fake CompoundUnitType.
        mock_unit_type_class = mock.Mock(spec=type)
        config.mock_unit_type.mock_add_spec(mock_unit_type_class)

        # Make another Unit to divide by.
        other_unit = mock.Mock(spec=UnitInterface)

        # Create a fake type for the other unit.
        other_type = mock.Mock()
        other_type_property = mock.PropertyMock(return_value=other_type)
        type(other_unit).type = other_type_property

        # Make it look like the types are incompatible.
        other_type.is_compatible.return_value = False

        # Act.
        quotient = config.div_unit / other_unit

        # Assert.
        # It should have checked compatibility.
        other_type.is_compatible.assert_called_once_with(config.mock_unit_type)
        # It should not have attempted to convert.
        config.mock_unit_type.assert_not_called()

        # It should have created a new compound unit.
        mock_unit_type_class.assert_called_once_with(
            Operation.DIV, config.mock_unit_type, other_type)

        compound_unit = mock_unit_type_class.return_value
        compound_unit.apply_to.assert_called_once_with(config.div_unit,
                                                       other_unit)

        # It should have returned the compound unit.
        assert quotient == compound_unit.apply_to.return_value

    def test_raw(self, config: UnitConfig) -> None:
        """
        Tests that we can successfully get the raw value.
        :param config: The configuration to use.
        """
        # Arrange.
        # Set reasonable raw values for the sub-units.
        mock_left_raw = mock.PropertyMock(return_value=42)
        mock_right_raw = mock.PropertyMock(return_value=7)
        type(config.mock_left_unit).raw = mock_left_raw
        type(config.mock_right_unit).raw = mock_right_raw

        # Act.
        quotient = config.div_unit.raw

        # Assert.
        # It should have divided the raw values.
        assert quotient == pytest.approx(6)

    def test_name(self, config: UnitConfig) -> None:
        """
        Tests that we can successfully get the name of the unit.
        :param config: The configuration to use.
        """
        # Arrange.
        # Set names for the sub-units.
        mock_left_name = mock.PropertyMock(return_value="m")
        mock_right_name = mock.PropertyMock(return_value="s")
        type(config.mock_left_unit).name = mock_left_name
        type(config.mock_right_unit).name = mock_right_name

        # Act.
        name = config.div_unit.name

        # Assert.
        assert name == "m\n-\ns"
