from typing import NamedTuple
import unittest.mock as mock

import pytest

from pyunits.compound_units import mul_unit
from pyunits.compound_units.compound_unit_type import CompoundUnitType
from pyunits.compound_units.operations import Operation
from pyunits import unit_base
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
        # Make sure that getting a new instance returns the mock.
        mock_unit_type.get.return_value = mock_unit_type
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

