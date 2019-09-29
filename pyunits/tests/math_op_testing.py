import unittest.mock as mock

import numpy as np

from pyunits.compound_units.compound_unit_type import  CompoundUnitType
from pyunits.compound_units.operations import Operation
from pyunits.types import Numeric
from pyunits.unit_interface import UnitInterface


def test_mul_incompatible_unit(unit: UnitInterface, mock_type: mock.Mock,
                               mock_mul_factory: mock.Mock,
                               mock_simplify: mock.Mock,
                               *,
                               simplify: bool,
                               passes_op: bool) -> None:
    """
    Tests the multiplication of a unit with another that is incompatible.
    :param unit: The unit to test the multiplication of.
    :param mock_type: The mocked UnitType for this unit.
    :param mock_mul_factory: The mocked CompoundUnitType to use for creating the
    resulting compound unit.
    :param mock_simplify: The mocked simplify() function.
    :param simplify: Whether to make it look like the resulting CompoundUnitType
    can be simplified.
    :param passes_op: Whether we expect the operation to be passed to the
    CompoundUnitType constructor or not.
    """
    # Arrange.
    # Make another Unit to multiply by.
    other_unit = mock.Mock(spec=UnitInterface)

    # Create a fake type for the other unit.
    other_type = mock.Mock()
    other_type_property = mock.PropertyMock(return_value=other_type)
    type(other_unit).type = other_type_property

    # Make it look like we can standardize the other unit and get a raw value.
    standard_other = other_unit.to_standard.return_value
    mock_raw = mock.PropertyMock(return_value=np.array(42.0))
    type(standard_other).raw = mock_raw

    # Make it look like the types are incompatible.
    other_type.is_compatible.return_value = False

    mock_simplified = mock.Mock(spec=CompoundUnitType)
    if simplify:
        # Make it look like simplification produced a different type.
        mock_simplify.side_effect = None
        mock_simplify.return_value = mock_simplified

    # Act.
    product = unit * other_unit

    # Assert.
    # It should have checked compatibility.
    other_type.is_compatible.assert_called_once_with(mock_type)
    # It should not have attempted to convert.
    mock_type.assert_not_called()

    # It should have created a new compound unit.
    if passes_op:
        mock_mul_factory.assert_called_once_with(
            Operation.MUL, mock_type, other_type
        )
    else:
        mock_mul_factory.assert_called_once_with(
            mock_type, other_type
        )

    # It should have applied the compound unit to the operands.
    compound_unit = mock_mul_factory.return_value
    # It should have simplified the unit type.
    mock_simplify.assert_called_once_with(compound_unit, mock.ANY)

    if not simplify:
        # It should have used the original type.
        compound_unit.apply_to.assert_called_once_with(unit, other_unit)

        # It should have returned the compound unit.
        assert product == compound_unit.apply_to.return_value

    else:
        # It should have used the simplified type.
        # The sub-units should have been standardized.
        other_unit.to_standard.assert_called_once_with()

        # It should have created the simplified unit.
        raw_product = unit.to_standard().raw * np.array(42.0)
        mock_simplified.assert_called_once()
        args, _ = mock_simplified.call_args
        np.testing.assert_array_almost_equal(args[0], raw_product)

        # It should have returned the compound unit.
        assert product == mock_simplified.return_value


def test_div_incompatible_unit(unit: UnitInterface, mock_type: mock.Mock,
                               mock_div_factory: mock.Mock,
                               mock_simplify: mock.Mock,
                               *,
                               simplify: bool,
                               passes_op: bool) -> None:
    """
    Tests the division of a unit by another that is incompatible.
    :param unit: The unit to test the division of.
    :param mock_type: The mocked UnitType for this unit.
    :param mock_div_factory: The mocked CompoundUnitType to use for creating the
    resulting compound unit.
    :param mock_simplify: The mocked simplify() function.
    :param simplify: Whether to make it look like the resulting CompoundUnitType
    can be simplified.
    :param passes_op: Whether we expect the operation to be passed to the
    CompoundUnitType constructor or not.
    """
    # Make another Unit to divide by.
    other_unit = mock.Mock(spec=UnitInterface)

    # Create a fake type for the other unit.
    other_type = mock.Mock()
    other_type_property = mock.PropertyMock(return_value=other_type)
    type(other_unit).type = other_type_property

    # Make it look like we can standardize the other unit and get a raw value.
    standard_other = other_unit.to_standard.return_value
    mock_raw = mock.PropertyMock(return_value=np.array(42.0))
    type(standard_other).raw = mock_raw

    # Make it look like the types are incompatible.
    other_type.is_compatible.return_value = False

    mock_simplified = mock.Mock(spec=CompoundUnitType)
    if simplify:
        # Make it look like simplification produced a different type.
        mock_simplify.side_effect = None
        mock_simplify.return_value = mock_simplified

    # Act.
    quotient = unit / other_unit

    # Assert.
    # It should have checked compatibility.
    other_type.is_compatible.assert_called_once_with(mock_type)
    # It should not have attempted to convert.
    mock_type.assert_not_called()

    # It should have created a new compound unit.
    if passes_op:
        mock_div_factory.assert_called_once_with(
            Operation.DIV, mock_type, other_type
        )
    else:
        mock_div_factory.assert_called_once_with(
            mock_type, other_type
        )

    compound_unit = mock_div_factory.return_value
    # It should have simplified the unit type.
    mock_simplify.assert_called_once_with(compound_unit, mock.ANY)

    if not simplify:
        # It should have used the original type.
        compound_unit.apply_to.assert_called_once_with(unit, other_unit)

        # It should have returned the compound unit.
        assert quotient == compound_unit.apply_to.return_value

    else:
        # It should have used the simplified type.
        # The sub-units should have been standardized.
        other_unit.to_standard.assert_called_once_with()

        # It should have created the simplified unit.
        raw_quotient = unit.to_standard().raw / np.array(42.0)
        mock_simplified.assert_called_once()
        args, _ = mock_simplified.call_args
        np.testing.assert_array_almost_equal(args[0], raw_quotient)

        # It should have returned the compound unit.
        assert quotient == mock_simplified.return_value


def test_mul_numeric(unit: UnitInterface, mock_type: mock.Mock,
                     multiplier: Numeric, expected_product: Numeric
                     ) -> None:
    """
    Tests the multiplication of a unit by a numeric value.
    :param unit: The unit to multiply.
    :param mock_type: The mocked corresponding UnitType.
    :param multiplier: The value to multiply by.
    :param expected_product: The expected raw product value.
    """
    # Act.
    # Multiply by a numeric value.
    product = unit * multiplier

    # Assert.
    # It should have created a new unit of the same type.
    assert mock_type.call_count == 1
    # It should have been created with the raw product value.
    my_args, _ = mock_type.call_args
    got_product = my_args[0]
    np.testing.assert_array_equal(expected_product, got_product)

    assert product == mock_type.return_value


def test_div_numeric(unit: UnitInterface, mock_type: mock.Mock,
                     divisor: Numeric, expected_quotient: Numeric
                     ) -> None:
    """
    Tests the division of a unit by a numeric value.
    :param unit: The unit to multiply.
    :param mock_type: The mocked corresponding UnitType.
    :param divisor: The value to divide by.
    :param expected_quotient: The expected raw quotient value.
    """
    # Act.
    # Divide by a numeric value.
    quotient = unit / divisor

    # Assert.
    # It should have created a new unit of the same type.
    assert mock_type.call_count == 1
    # It should have been created with the raw product value.
    my_args, _ = mock_type.call_args
    got_quotient = my_args[0]
    np.testing.assert_array_almost_equal(expected_quotient, got_quotient)

    assert quotient == mock_type.return_value
