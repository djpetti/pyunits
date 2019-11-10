from typing import NamedTuple
import enum
import unittest.mock as mock

import numpy as np

import pytest

from pyunits.compound_units.compound_unit import CompoundUnit
from pyunits.compound_units.operations import Operation
from pyunits.compound_units import compound_unit_type
from pyunits.exceptions import UnitError
from pyunits.tests.testing_types import UnitTypeFactory
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
        :param mock_compound_unit: The mock CompoundUnit subclass that the units
        we create will be instances of.
        """
        compound_type: compound_unit_type.CompoundUnitType
        mock_left_sub_type: mock.Mock
        mock_right_sub_type: mock.Mock
        mock_compound_unit: mock.Mock

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
    @pytest.fixture(params=[Operation.MUL, Operation.DIV])
    def config(cls, request: RequestType) -> UnitConfig:
        """
        Generates configuration for the tests.
        :param request: The request object to use for parametrization.
        :return: The UnitConfig that it created.
        """
        # The operation that we want to perform.
        operation = request.param

        class LeftType(UnitType):

            def __init__(self):
                # This is an ugly hack to work around an idiosyncrasy of
                # Python's mock library: If we try to pass UnitType
                # sub-classes as specs, then Python calls __init__ when we
                # invoke the mock, instead of __call__, which is what we want.
                # We can induce the correct behavior by passing an instance as
                # the spec instead of the class, but I don't want this test to
                # depend on the actual UnitType.__init__ method. Hence, the
                # stubbing-out of __init__ in the subclass.
                pass

        class RightType(UnitType):

            def __init__(self):
                pass

        left_sub_type = mock.MagicMock(spec=LeftType())
        right_sub_type = mock.MagicMock(spec=RightType())

        # Make it look like the two types are not compatible with each-other,
        # otherwise CompoundUnitType will yell at us.
        left_sub_type.is_compatible.return_value = False
        right_sub_type.is_compatible.return_value = False

        # Stub out the CompoundUnit constructors.
        mock_mul_unit = mock.Mock()
        mock_div_unit = mock.Mock()
        compound_unit_type.CompoundUnitType.OPERATION_TO_CLASS = \
            {Operation.MUL: mock_mul_unit, Operation.DIV: mock_div_unit}

        # Ensure that it creates a new unit every time.
        compound_unit_type.CompoundUnitType.clear_interning_cache()

        # Create the compound type to test with.
        compound_type = compound_unit_type.CompoundUnitType\
            .get(operation, left_sub_type, right_sub_type)

        mock_compound_unit = \
            compound_unit_type.CompoundUnitType.OPERATION_TO_CLASS[operation]
        return cls.UnitConfig(compound_type=compound_type,
                              mock_left_sub_type=left_sub_type,
                              mock_right_sub_type=right_sub_type,
                              mock_compound_unit=mock_compound_unit)

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

        # It should have created the CompoundUnit.
        config.mock_compound_unit.assert_called_once_with(config.compound_type,
                                                          converted_left,
                                                          converted_right)

        # It should have returned it.
        assert compound_unit == config.mock_compound_unit.return_value

    @pytest.mark.parametrize("value", [10, 5.0, (1, 2, 3), np.array([4, 5])])
    def test_call_raw(self, config: UnitConfig, value: UnitValue) -> None:
        """
        Tests that we can create a CompoundUnit directly from a raw value.
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

        # It should have created the CompoundUnit.
        config.mock_compound_unit.assert_called_once_with(config.compound_type,
                                                          left_unit, right_unit)

        # It should have returned it.
        assert compound_unit == config.mock_compound_unit.return_value

    def test_call_other(self, config: UnitConfig) -> None:
        """
        Tests that we can create a CompoundUnit directly from another one.
        :param config: The configuration to use for the test.
        """
        # Arrange.
        # Create an existing unit.
        template_unit = mock.Mock(spec=CompoundUnit)
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

        # It should have created the CompoundUnit.
        config.mock_compound_unit.assert_called_once_with(config.compound_type,
                                                          converted_left,
                                                          converted_right)

        # It should have returned it.
        assert compound_unit == config.mock_compound_unit.return_value

    def test_call_other_incompatible(self, config: UnitConfig) -> None:
        """
        Tests that creating a CompoundUnit from another unit fails when the unit
        has an incompatible type.
        :param config: The configuration to use.
        """
        # Arrange.
        # Create an existing unit.
        template_unit = mock.Mock(spec=CompoundUnit)

        # Mock the type so that it is incompatible.
        mock_type_property = mock.PropertyMock()
        type(template_unit).type = mock_type_property

        # Act and assert.
        with pytest.raises(UnitError):
            config.compound_type(template_unit)

    def test_standard_unit_class(self, config: UnitConfig,
                                 unit_type_factory: UnitTypeFactory) -> None:
        """
        Tests that standard_unit_class() works.
        :param config: The configuration to use.
        :param unit_type_factory: The factory to use for creating new UnitTypes.
        """
        # Arrange.
        # Make it look like we have valid standard classes for both our
        # subtypes.
        mock_standard_left = unit_type_factory("StandardLeftType")
        mock_standard_right = unit_type_factory("StandardRightType")
        config.mock_left_sub_type.standard_unit_class.return_value = \
            mock_standard_left
        config.mock_right_sub_type.standard_unit_class.return_value = \
            mock_standard_right

        # Make it look like they're incompatible so that the CompoundUnitType
        # constructor doesn't complain.
        mock_standard_left.is_compatible.return_value = False
        mock_standard_right.is_compatible.return_value = False

        # Act.
        got_standard = config.compound_type.standard_unit_class()

        # Assert.
        # It should have standardized the sub-types.
        config.mock_left_sub_type.standard_unit_class.assert_called_once_with()
        config.mock_right_sub_type.standard_unit_class.assert_called_once_with()

        # It should have created the new CompoundUnitType.
        assert got_standard.left == mock_standard_left
        assert got_standard.right == mock_standard_right
        assert got_standard.operation == config.compound_type.operation

    @pytest.mark.parametrize("reverse_sub_units", [False, True])
    def test_is_compatible_subunits(self, config: UnitConfig,
                                    reverse_sub_units: bool) -> None:
        """
        Tests that is_compatible can determine when two UnitTypes are compatible
        with various sub-unit configurations.
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
        compare_type.operation = config.compound_type.operation

        # Act.
        is_compatible = config.compound_type.is_compatible(compare_type)

        # Assert.
        # It should have checked sub-unit compatibility.
        compare_type.left.is_compatible.assert_called()
        if config.compound_type.operation == Operation.MUL:
            compare_type.right.is_compatible.assert_called()

        if config.compound_type.operation != Operation.MUL \
                and reverse_sub_units:
            # In this case, the operation is not commutative and the sub-units
            # are reversed, so this should be flagged as incompatible.
            assert not is_compatible
        else:
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
            compare_type.operation = -1
        else:
            compare_type.operation = config.compound_type.operation

        # Act.
        is_compatible = config.compound_type.is_compatible(compare_type)

        # Assert.
        # They should not be compatible.
        assert not is_compatible

    @pytest.mark.parametrize("operation", [Operation.MUL, Operation.DIV])
    def test_init_compatible(self, config: UnitConfig,
                             operation: Operation) -> None:
        """
        Tests that creating a new CompoundUnitType fails when the sub-types are
        compatible with each-other.
        :param config: The configuration to use for the test.
        :param operation: The operation to use for the test.
        """
        # Arrange.
        compound_unit_type.CompoundUnitType.clear_interning_cache()

        # Make it look like the sub-types are compatible.
        config.mock_right_sub_type.is_compatible.return_value = True
        config.mock_left_sub_type.is_compatible.return_value = True

        # Act and assert.
        with pytest.raises(UnitError, match=r"compatible"):
            compound_unit_type.CompoundUnitType\
                .get(operation, config.mock_left_sub_type,
                     config.mock_right_sub_type)

        # It should have checked the compatibility.
        num_checks = 0
        num_checks += config.mock_left_sub_type.is_compatible.call_count
        num_checks += config.mock_right_sub_type.is_compatible.call_count
        assert num_checks > 0

    def test_init_squared(self, config: UnitConfig) -> None:
        """
        Tests that it doesn't complain when we try to create a unit squared
        compound unit.
        :param config: The configuration to use for the test.
        """
        # Arrange.
        compound_unit_type.CompoundUnitType.clear_interning_cache()

        # Act.
        # Create a squared CompoundUnitType.
        compound_unit_type.CompoundUnitType\
            .get(Operation.MUL, config.mock_left_sub_type,
                 config.mock_left_sub_type)

        # Assert.
        # It should have checked the compatibility.
        num_checks = 0
        num_checks += config.mock_left_sub_type.is_compatible.call_count
        num_checks += config.mock_right_sub_type.is_compatible.call_count
        assert num_checks > 0

    def test_interning(self, config: UnitConfig) -> None:
        """
        Tests that interning works correctly.
        :param config: The configuration to use for the test.
        """
        # Arrange.
        operation = config.compound_type.operation

        # Act.
        type2 = compound_unit_type.CompoundUnitType\
            .get(operation, config.mock_left_sub_type,
                 config.mock_left_sub_type)
        # Should be the same as 1.
        type3 = compound_unit_type.CompoundUnitType\
            .get(operation, config.mock_left_sub_type,
                 config.mock_right_sub_type)
        # Should also be the same as 1 if it is a product.
        type4 = compound_unit_type.CompoundUnitType\
            .get(operation, config.mock_right_sub_type,
                 config.mock_left_sub_type)

        # Assert.
        # The second type should always be new.
        assert config.compound_type != type2
        # The third one should always be the same as the first.
        assert type3 == config.compound_type
        # The fourth should only be the same if it is a product, and thus the
        # argument order doesn't matter.
        if operation == Operation.MUL:
            assert type4 == config.compound_type
        else:
            assert type4 != config.compound_type
