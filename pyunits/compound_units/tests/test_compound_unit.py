from typing import NamedTuple, Type
import unittest.mock as mock

import pytest

from pyunits.compound_units.compound_unit import CompoundUnit
from pyunits.compound_units.compound_unit_type import CompoundUnitType
from pyunits.compound_units.div_unit import DivUnit
from pyunits.compound_units.mul_unit import MulUnit
from pyunits.types import RequestType
from pyunits.unit_interface import UnitInterface


class TestCompoundUnit:
    """
    Unified tests for all CompoundUnit subclasses.
    """

    class UnitConfig(NamedTuple):
        """
        Represents configuration for the tests.
        :param compound_unit: The CompoundUnit under test.
        :param mock_unit_type: The mocked type of the MulUnit.
        :param mock_left_unit: The mocked left sub-unit.
        :param mock_right_unit: The mocked right sub-unit.
        """
        compound_unit: CompoundUnit
        mock_unit_type: mock.Mock
        mock_left_unit: mock.Mock
        mock_right_unit: mock.Mock

    class ClassSpecificConfig(NamedTuple):
        """
        Encapsulates configuration that is specific to the particular sub-class
        that we are testing.
        :param class_under_test: The sub-class that is being tested.
        """
        class_under_test: Type

    # Class-specific configuration for each subclass.
    _MUL_UNIT_CONFIG = ClassSpecificConfig(class_under_test=MulUnit)
    _DIV_UNIT_CONFIG = ClassSpecificConfig(class_under_test=DivUnit)

    @classmethod
    @pytest.fixture(params=[_MUL_UNIT_CONFIG, _DIV_UNIT_CONFIG])
    def class_specific_config(cls, request: RequestType) -> ClassSpecificConfig:
        """
        Fixture that produces the class-specific configuration for each
        subclass.
        :param request: The request object to use for parametrization.
        :return: The class-specific configuration for the subclass we are
        testing.
        """
        return request.param

    @classmethod
    @pytest.fixture
    def config(cls, class_specific_config: ClassSpecificConfig) -> UnitConfig:
        """
        Creates new configuration for a test.
        :param class_specific_config: Configuration that is specific to the
        subclass that we are testing.
        :return: The configuration that it created,
        """
        # Create the fake unit type.
        mock_unit_type = mock.Mock(spec=CompoundUnitType)
        # Create the fake sub-units.
        mock_left_unit = mock.Mock(spec=UnitInterface)
        mock_right_unit = mock.Mock(spec=UnitInterface)

        # Create the fake unit.
        my_class = class_specific_config.class_under_test
        my_compound_unit = my_class(mock_unit_type, mock_left_unit,
                                    mock_right_unit)

        return cls.UnitConfig(compound_unit=my_compound_unit,
                              mock_unit_type=mock_unit_type,
                              mock_left_unit=mock_left_unit,
                              mock_right_unit=mock_right_unit)

    def test_to_standard(self, config: UnitConfig) -> None:
        """
        Tests that to_standard() works.
        :param config: The configuration to use.
        """
        # Arrange done in fixtures.
        # Act.
        standard_unit = config.compound_unit.to_standard()

        # Assert.
        # It should have standardized both of the sub-units.
        config.mock_left_unit.to_standard.assert_called_once_with()
        config.mock_right_unit.to_standard.assert_called_once_with()

        # It should have re-applied the compound unit to the standardized
        # sub-units.
        standard_left = config.mock_left_unit.to_standard.return_value
        standard_right = config.mock_right_unit.to_standard.return_value
        config.mock_unit_type.apply_to.assert_called_once_with(standard_left,
                                                               standard_right)

        # It should have returned the new DivUnit.
        assert standard_unit == config.mock_unit_type.apply_to.return_value

    def test_cast_to(self, config: UnitConfig) -> None:
        """
        Tests that cast_to() works.
        :param config: The configuration to use.
        """
        # Arrange.
        # Create a unit to cast to.
        cast_to_type = mock.Mock(spec=CompoundUnitType)

        # Act.
        casted = config.compound_unit.cast_to(cast_to_type)

        # Assert.
        # It should have casted the left and right sub-units individually.
        config.mock_left_unit.cast_to.assert_called_once_with(cast_to_type.left)
        config.mock_right_unit.cast_to.assert_called_once_with(
            cast_to_type.right)

        # It should have applied the compound unit to the casted sub-units.
        left_casted = config.mock_left_unit.cast_to.return_value
        right_casted = config.mock_right_unit.cast_to.return_value
        cast_to_type.apply_to.assert_called_once_with(left_casted, right_casted)

        # It should have returned the result.
        assert casted == cast_to_type.apply_to.return_value

    def test_left(self, config: UnitConfig) -> None:
        """
        Tests that getting the left sub-unit works.
        :param config: The configuration to use.
        """
        # Arrange done in fixtures.
        # Act and assert.
        assert config.compound_unit.left == config.mock_left_unit

    def test_right(self, config: UnitConfig) -> None:
        """
        Tests that getting the right sub-unit works.
        :param config: The configuration to use.
        """
        # Arrange done in fixtures.
        # Act and assert.
        assert config.compound_unit.right == config.mock_right_unit
