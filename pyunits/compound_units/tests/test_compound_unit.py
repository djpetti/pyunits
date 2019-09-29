from typing import Callable, NamedTuple, Type
import unittest.mock as mock

import numpy as np

import pytest

from pyunits import unit_base
from pyunits.compound_units.compound_unit import CompoundUnit
from pyunits.compound_units.compound_unit_type import CompoundUnitType
from pyunits.compound_units.div_unit import DivUnit
from pyunits.compound_units.mul_unit import MulUnit
from pyunits.tests import math_op_testing
from pyunits.types import RequestType, UnitValue
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
        :param mock_simplify: The mocked simplify function.
        """
        compound_unit: CompoundUnit
        mock_unit_type: mock.Mock
        mock_left_unit: mock.Mock
        mock_right_unit: mock.Mock
        mock_simplify: mock.Mock

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
        # Make sure the get() method returns the same mock instance.
        mock_unit_type.get.return_value = mock_unit_type

        # Create the fake sub-units.
        mock_left_unit = mock.Mock(spec=UnitInterface)
        mock_right_unit = mock.Mock(spec=UnitInterface)

        # Create the fake unit.
        my_class = class_specific_config.class_under_test
        my_compound_unit = my_class(mock_unit_type, mock_left_unit,
                                    mock_right_unit)

        with mock.patch(unit_base.__name__ + ".unit_analysis.simplify") as \
                mock_simplify:
            # Make it look like it never simplifies anything.
            mock_simplify.side_effect = lambda x, _: x

            yield cls.UnitConfig(compound_unit=my_compound_unit,
                                 mock_unit_type=mock_unit_type,
                                 mock_left_unit=mock_left_unit,
                                 mock_right_unit=mock_right_unit,
                                 mock_simplify=mock_simplify)

            # Finalization done upon exit from context manager.

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

    @pytest.mark.parametrize("mul_by", [10, 5.0, np.array([1, 2, 3])])
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

        if isinstance(config.compound_unit, MulUnit):
            expected_product = 2 * 3 * mul_by
        else:
            expected_product = 2 / 3 * mul_by

        # Act and assert.
        math_op_testing.test_mul_numeric(config.compound_unit,
                                         config.mock_unit_type,
                                         mul_by,
                                         expected_product)

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

        if isinstance(config.compound_unit, MulUnit):
            expected_quotient = left_sub_value * right_sub_value / div_by
        else:
            expected_quotient = left_sub_value / right_sub_value / div_by

        # Act and assert.
        math_op_testing.test_div_numeric(config.compound_unit,
                                         config.mock_unit_type,
                                         div_by,
                                         expected_quotient)

    @pytest.mark.parametrize("test_func",
                             [math_op_testing.test_mul_incompatible_unit,
                              math_op_testing.test_div_incompatible_unit])
    @pytest.mark.parametrize("simplify", [False, True])
    def test_incompatible_unit(self, config: UnitConfig,
                               test_func: Callable,
                               simplify: bool) -> None:
        """
        Tests that arithmetic operations work correctly when the operands have
        incompatible types.
        :param config: The configuration to use.
        :param test_func: The function to use for performing the test.
        :param simplify: Whether we should simulate simplification of the
        resulting type.
        """
        # Arrange.
        # Make it look like the result of to_standard() has an actual value.
        standardized = config.mock_unit_type.apply_to.return_value
        mock_raw = mock.PropertyMock(return_value=np.array(5.0))
        type(standardized).raw = mock_raw

        # Act and assert.
        test_func(config.compound_unit, config.mock_unit_type,
                  config.mock_unit_type.get, config.mock_simplify,
                  simplify=simplify,
                  passes_op=True)
