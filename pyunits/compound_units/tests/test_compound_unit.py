from typing import NamedTuple, Type
import unittest.mock as mock

import pytest

from pyunits.compound_units import compound_unit
from pyunits.compound_units.compound_unit_type import CompoundUnitType
from pyunits.compound_units.div_unit import DivUnit
from pyunits.compound_units.mul_unit import MulUnit
from pyunits.compound_units.operations import Operation
from pyunits.tests.testing_types import UnitFactory
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
        :param mock_unit_type: The mocked type of the CompoundUnit.
        :param mock_left_unit: The mocked left sub-unit.
        :param mock_right_unit: The mocked right sub-unit.
        :param mock_do_mul: The mocked do_mul() function.
        :param mock_do_div: The mocked do_div() function.
        :param mock_do_add: The mocked do_add() function.
        :param mock_simplify: The mocked simplify() function.
        """
        compound_unit: compound_unit.CompoundUnit
        mock_unit_type: mock.Mock
        mock_left_unit: mock.Mock
        mock_right_unit: mock.Mock
        mock_do_mul: mock.Mock
        mock_do_div: mock.Mock
        mock_do_add: mock.Mock
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
    @pytest.fixture(params=[_MUL_UNIT_CONFIG, _DIV_UNIT_CONFIG],
                    ids=["mul_unit", "div_unit"])
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

        with mock.patch(compound_unit.__name__ + ".do_mul") as \
                mock_do_mul, \
                mock.patch(compound_unit.__name__ + ".do_div") as \
                mock_do_div, \
                mock.patch(compound_unit.__name__ + ".do_add") as \
                mock_do_add, \
                mock.patch(compound_unit.__name__ + ".simplify") as \
                mock_simplify:
            yield cls.UnitConfig(compound_unit=my_compound_unit,
                                 mock_unit_type=mock_unit_type,
                                 mock_left_unit=mock_left_unit,
                                 mock_right_unit=mock_right_unit,
                                 mock_do_mul=mock_do_mul,
                                 mock_do_div=mock_do_div,
                                 mock_do_add=mock_do_add,
                                 mock_simplify=mock_simplify)

            # Finalization done upon exit from context manager.

    @pytest.mark.parametrize(["left_standard", "right_standard",
                              "compound_standard"],
                             [(True, True, True), (False, False, False),
                              (True, False, False), (False, True, False)])
    def test_is_standard(self, config: UnitConfig,
                         left_standard: bool, right_standard: bool,
                         compound_standard: bool) -> None:
        """
        Tests that is_standard() works.
        :param config: The configuration to use.
        :param left_standard: Whether the left unit is standard.
        :param right_standard: Whether the right unit is standard.
        :param compound_standard: Whether the compound unit should be standard.
        """
        # Arrange.
        config.mock_left_unit.is_standard.return_value = left_standard
        config.mock_right_unit.is_standard.return_value = right_standard

        # Act.
        got_standard = config.compound_unit.is_standard()

        # Assert.
        assert got_standard == compound_standard

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

        # It should have gotten the standard type.
        config.mock_unit_type.standard_unit_class.assert_called_once_with()
        standard_type = config.mock_unit_type.standard_unit_class.return_value

        # It should have re-applied the compound unit to the standardized
        # sub-units.
        standard_left = config.mock_left_unit.to_standard.return_value
        standard_right = config.mock_right_unit.to_standard.return_value
        standard_type.apply_to.assert_called_once_with(standard_left,
                                                       standard_right)
        standard_applied = standard_type.apply_to.return_value

        # It should have simplified the result.
        config.mock_simplify.assert_called_once_with(standard_applied, mock.ANY)

        # It should have returned the simplified CompoundUnit.
        assert standard_unit == config.mock_simplify.return_value

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

    def test_operation(self, config: UnitConfig) -> None:
        """
        Tests that getting the operation works.
        :param config: The configuration to use.
        """
        # Arrange.
        # Mock the operation from the type.
        mock_operation = mock.PropertyMock(return_value=Operation.MUL)
        type(config.mock_unit_type).operation = mock_operation

        # Act.
        got_operation = config.compound_unit.operation

        # Assert.
        assert got_operation == Operation.MUL

    @pytest.mark.parametrize("is_reversed", [False, True],
                             ids=["normal", "reversed"])
    def test_mul(self, config: UnitConfig, unit_factory: UnitFactory,
                 is_reversed: bool) -> None:
        """
        Tests that we can multiply the compound unit by another.
        :param config: The configuration to use for testing.
        :param unit_factory: The UnitFactory to use for creating test units.
        :param is_reversed: Whether to use reversed multiplication for the test.
        """
        # Arrange.
        # Create a fake unit to multiply by.
        mul_by = unit_factory("Unit1", 1.0)

        # Act.
        if not is_reversed:
            product = config.compound_unit * mul_by
        else:
            product = mul_by * config.compound_unit

        # Assert.
        # It should do the same thing either way because for multiplication,
        # order doesn't matter.
        config.mock_do_mul.assert_called_once_with(mock.ANY,
                                                   config.compound_unit,
                                                   mul_by)
        assert product == config.mock_do_mul.return_value

    @pytest.mark.parametrize("is_reversed", [False, True],
                             ids=["normal", "reversed"])
    def test_div(self, config: UnitConfig, unit_factory: UnitFactory,
                 is_reversed: bool) -> None:
        """
        Tests that we can divide the compound unit by another.
        :param config: The configuration to use for testing.
        :param unit_factory: The UnitFactory to use for creating test units.
        :param is_reversed: Whether to use reversed multiplication for the test.
        """
        # Arrange.
        # Create a fake unit to divide by.
        div_by = unit_factory("Unit1", 1.0)

        # Act.
        if not is_reversed:
            quotient = config.compound_unit / div_by
            do_div_args = (config.compound_unit, div_by)
        else:
            quotient = div_by / config.compound_unit
            do_div_args = (div_by, config.compound_unit)

        # Assert.
        config.mock_do_div.assert_called_once_with(mock.ANY,
                                                   *do_div_args)
        assert quotient == config.mock_do_div.return_value

    @pytest.mark.parametrize("is_reversed", [False, True],
                             ids=["normal", "reversed"])
    def test_add(self, config: UnitConfig, unit_factory: UnitFactory,
                 is_reversed: bool) -> None:
        """
        Tests that we can add the unit to another.
        :param config: The configuration to use for testing.
        :param unit_factory: The UnitFactory to use for creating test units.
        :param is_reversed: Whether to use reversed multiplication for the test.
        """
        # Arrange.
        # Create a fake unit to add.
        add_to = mock.Mock(spec=compound_unit.CompoundUnit)

        # Act.
        if not is_reversed:
            unit_sum = config.compound_unit + add_to
        else:
            unit_sum = add_to + config.compound_unit

        # Assert.
        # It should do the same thing either way because for addition,
        # order doesn't matter.
        config.mock_do_add.assert_called_once_with(config.compound_unit,
                                                   add_to)
        assert unit_sum == config.mock_do_add.return_value
