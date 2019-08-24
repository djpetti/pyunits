from typing import NamedTuple
import enum
import unittest.mock as mock

import numpy as np

import pytest

from pyunits.compound_units.mul_unit import MulUnit
from pyunits.compound_units.operations import Operation
from pyunits.compound_units import compound_unit_type
from pyunits.exceptions import UnitError
from pyunits.types import RequestType, UnitValue
from pyunits.unit_interface import UnitInterface
from pyunits.unit_type import UnitType


class TestCompoundUnitType:
    """
    Tests for the CompoundUnitType class.
    """

    class UnitConfig(NamedTuple):
        """
        Encapsulates common configuration for tests.
        :param compound_type: The CompoundUnitType under test.
        :param mock_left_sub_type: The UnitType of the left operand.
        :param mock_right_sub_type: The UnitType of the right operand.
        :param mock_mul_unit: The mock MulUnit constructor.
        """
        compound_type: compound_unit_type.CompoundUnitType
        mock_left_sub_type: mock.Mock
        mock_right_sub_type: mock.Mock
        mock_mul_unit: mock.Mock

    class TypeIncompatibilities(enum.IntEnum):
        """
        Enumerates ways in which UnitTypes can be incompatible with a
        CompoundUnitType.
        """
        # The other type is not a CompoundUnitType.
        NOT_COMPOUND = enum.auto()
        # The other type does not have compatible sub-units.
        INCOMPATIBLE_SUB_UNITS = enum.auto()
        # The other type does not implement the same operation.
        DIFFERENT_OPERATIONS = enum.auto()

    @classmethod
    @pytest.fixture(params=[Operation.MUL])
    def config(cls, request: RequestType) -> UnitConfig:
        """
        Generates configuration for the tests.
        :param request: The request object to use for parametrization.
        :return: The UnitConfig that it created.
        """
        # The operation that we want to perform.
        operation = request.param

        # Create the left and right unit types.
        left_sub_type = mock.MagicMock(spec=UnitType)
        right_sub_type = mock.MagicMock(spec=UnitType)

        # Mock the name attribute so logging works.
        left_sub_type.__name__ = "LeftType"
        right_sub_type.__name__ = "RightType"

        with mock.patch(compound_unit_type.__name__ + ".MulUnit", spec=True) \
                as mock_mul_unit:
            # Create the compound type to test with.
            compound_type = compound_unit_type.CompoundUnitType(operation,
                                                                left_sub_type,
                                                                right_sub_type)

            yield cls.UnitConfig(compound_type=compound_type,
                                 mock_left_sub_type=left_sub_type,
                                 mock_right_sub_type=right_sub_type,
                                 mock_mul_unit=mock_mul_unit)

            # Finalization done implicitly upon exit from context manager.

    def test_left_right(self, config: UnitConfig) -> None:
        """
        Tests that we can get the left and right sub-units.
        :param config: The configuration to use for the test.
        """
        # Arrange done in fixtures.
        # Act.
        left = config.compound_type.left
        right = config.compound_type.right

        # Assert.
        assert left == config.mock_left_sub_type
        assert right == config.mock_right_sub_type

    def test_apply_to(self, config: UnitConfig) -> None:
        """
        Tests that we can create a new Unit instance using the apply_to()
        method.
        :param config: The configuration to use for the test.
        """
        # Arrange.
        # Create fake sub-units to use.
        mock_left_unit = mock.Mock(spec=UnitInterface)
        mock_right_unit = mock.Mock(spec=UnitInterface)

        # Act.
        compound_unit = config.compound_type.apply_to(mock_left_unit,
                                                      mock_right_unit)

        # Assert.
        # It should have converted the sub-units.
        config.mock_left_sub_type.assert_called_once_with(mock_left_unit)
        config.mock_right_sub_type.assert_called_once_with(mock_right_unit)
        converted_left = config.mock_left_sub_type.return_value
        converted_right = config.mock_right_sub_type.return_value

        # It should have created the MulUnit.
        config.mock_mul_unit.assert_called_once_with(config.compound_type,
                                                     converted_left,
                                                     converted_right)

        # It should have returned it.
        assert compound_unit == config.mock_mul_unit.return_value

    @pytest.mark.parametrize("value", [10, 5.0, (1, 2, 3), np.array([4, 5])])
    def test_call_raw(self, config: UnitConfig, value: UnitValue) -> None:
        """
        Tests that we can create a MulUnit directly from a raw value.
        :param config: The configuration to use for the test.
        :param value: The raw value to use.
        """
        # Arrange done in fixtures.
        # Act.
        compound_unit = config.compound_type(value)

        # Assert.
        # It should have initialized the sub-units.
        config.mock_left_sub_type.assert_called_once_with(value)
        config.mock_right_sub_type.assert_called_once_with(1)
        left_unit = config.mock_left_sub_type.return_value
        right_unit = config.mock_right_sub_type.return_value

        # It should have created the MulUnit.
        config.mock_mul_unit.assert_called_once_with(config.compound_type,
                                                     left_unit, right_unit)

        # It should have returned it.
        assert compound_unit == config.mock_mul_unit.return_value

    def test_call_other(self, config: UnitConfig) -> None:
        """
        Tests that we can create a MulUnit directly from another one.
        :param config: The configuration to use for the test.
        """
        # Arrange.
        # Create an existing unit.
        template_unit = mock.Mock(spec=MulUnit)
        template_left = template_unit.left
        template_right = template_unit.right

        # Mock the type so that it is compatible.
        mock_type_property = mock.PropertyMock(
            return_value=config.compound_type)
        type(template_unit).type = mock_type_property
        # Make it look like the sub-units are compatible.
        config.mock_left_sub_type.is_compatible.return_value = True
        config.mock_right_sub_type.is_compatible.return_value = True

        # Act.
        compound_unit = config.compound_type(template_unit)

        # Assert.
        # It should have verified compatibility.
        config.mock_left_sub_type.is_compatible.assert_called()
        config.mock_right_sub_type.is_compatible.assert_called()

        # It should have converted the sub-units.
        config.mock_left_sub_type.assert_called_once_with(template_left)
        config.mock_right_sub_type.assert_called_once_with(template_right)
        converted_left = config.mock_left_sub_type.return_value
        converted_right = config.mock_right_sub_type.return_value

        # It should have created the MulUnit.
        config.mock_mul_unit.assert_called_once_with(config.compound_type,
                                                     converted_left,
                                                     converted_right)

        # It should have returned it.
        assert compound_unit == config.mock_mul_unit.return_value

    def test_call_other_incompatible(self, config: UnitConfig) -> None:
        """
        Tests that creating a MulUnit from another unit fails when the unit
        has an incompatible type.
        :param config: The configuration to use.
        """
        # Arrange.
        # Create an existing unit.
        template_unit = mock.Mock(spec=MulUnit)

        # Mock the type so that it is incompatible.
        mock_type_property = mock.PropertyMock()
        type(template_unit).type = mock_type_property

        # Act and assert.
        with pytest.raises(UnitError):
            config.compound_type(template_unit)

    @pytest.mark.parametrize("reverse_sub_units", [False, True])
    def test_is_compatible(self, config: UnitConfig,
                           reverse_sub_units: bool) -> None:
        """
        Tests that is_compatible can determine that two UnitTypes are compatible
        when they are.
        :param config: The configuration to use.
        :param reverse_sub_units: If true, the sub-units will be reversed in the
        unit we are checking, which should technically still be regarded as the
        same unit since multiplication is commutative.
        """
        # Arrange.
        # Create a fake UnitType to check the compatibility of.
        compare_type = mock.Mock(spec=compound_unit_type.CompoundUnitType)

        left_type = config.compound_type.left
        right_type = config.compound_type.right

        # Make sure the sub-units are compatible.
        left_compatible_with = left_type
        right_compatible_with = right_type
        if reverse_sub_units:
            left_compatible_with = right_type
            right_compatible_with = left_type

        compare_type.left.is_compatible.side_effect = \
            lambda x: x == left_compatible_with
        compare_type.right.is_compatible.side_effect = \
            lambda x: x == right_compatible_with

        # Make sure the operations are the same.
        compare_type.operation = Operation.MUL

        # Act.
        is_compatible = config.compound_type.is_compatible(compare_type)

        # Assert.
        # It should have checked sub-unit compatibility.
        compare_type.left.is_compatible.assert_called()
        compare_type.right.is_compatible.assert_called()

        # They should be compatible.
        assert is_compatible

    @pytest.mark.parametrize("incompatibility",
                             [TypeIncompatibilities.NOT_COMPOUND,
                              TypeIncompatibilities.INCOMPATIBLE_SUB_UNITS,
                              TypeIncompatibilities.DIFFERENT_OPERATIONS])
    def test_is_compatible_not(self, config: UnitConfig,
                               incompatibility: TypeIncompatibilities) -> None:
        """
        Tests that is_compatible() properly marks two unit types as incompatible
        when this is so.
        :param config: The configuration to use.
        :param incompatibility: The type of incompatibility to test.
        """
        # Arrange.
        # Create a fake UnitType to check the compatibility of.
        if incompatibility == self.TypeIncompatibilities.NOT_COMPOUND:
            # Make it look like the other type is not compound.
            compare_type = mock.Mock()
        else:
            compare_type = mock.Mock(spec=compound_unit_type.CompoundUnitType)

        if incompatibility == self.TypeIncompatibilities.INCOMPATIBLE_SUB_UNITS:
            # Make it look like the sub-units are not compatible.
            compare_type.left.is_compatible.return_value = False
            compare_type.right.is_compatible.return_value = False
        else:
            compare_type.left.is_compatible.return_value = True
            compare_type.right.is_compatible.return_value = True

        if incompatibility == self.TypeIncompatibilities.DIFFERENT_OPERATIONS:
            # Make it look like the operations are different.
            compare_type.operation = Operation.DIV
        else:
            compare_type.operation = Operation.MUL

        # Act.
        is_compatible = config.compound_type.is_compatible(compare_type)

        # Assert.
        # They should not be compatible.
        assert not is_compatible
