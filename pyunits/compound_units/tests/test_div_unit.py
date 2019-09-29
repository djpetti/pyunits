from typing import NamedTuple
import unittest.mock as mock

import pytest

from pyunits.compound_units import div_unit
from pyunits.compound_units.compound_unit_type import CompoundUnitType
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
        # Make sure that getting a new instance returns the mock.
        mock_unit_type.get.return_value = mock_unit_type
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

    def test_div_compatible_unit(self, config: UnitConfig) -> None:
        """
        Tests that the division operation works correctly when a unit is
        divided by another with a compatible type.
        :param config: The configuration to use.
        """
        # Arrange.
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
