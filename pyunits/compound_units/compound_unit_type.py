from typing import cast

from loguru import logger

from ..exceptions import UnitError
from ..types import UnitValue
from ..unit_interface import UnitInterface
from ..unit_type import UnitType
from .mul_unit import MulUnit
from .operations import Operation


class CompoundUnitType(UnitType):
    """
    Unit type that represents the multiplication of two units.
    """

    def __init__(self, operation: Operation,
                 left_unit_class: UnitType, right_unit_class: UnitType):
        """
        :param operation: The operation performed by the compound unit.
        :param left_unit_class: The class of the first unit to multiply.
        :param right_unit_class: The class of the second unit to multiply.
        """
        self.__operation = operation
        self.__left_unit_class = left_unit_class
        self.__right_unit_class = right_unit_class

        logger.debug("Creating new unit type {} with sub-units {} and {}.",
                     operation, left_unit_class.__name__,
                     right_unit_class.__name__)

        # Functionally, the class we're "wrapping" is MulUnit.
        super().__init__(MulUnit)

    @property
    def left(self) -> UnitType:
        """
        :return: The first UnitType to multiply.
        """
        return self.__left_unit_class

    @property
    def right(self) -> UnitType:
        """
        :return: The second UnitType to multiply.
        """
        return self.__right_unit_class

    @property
    def operation(self) -> Operation:
        """
        :return: The operation being applied.
        """
        return self.__operation

    def apply_to(self, left_unit: UnitInterface,
                 right_unit: UnitInterface) -> MulUnit:
        """
        Applies the multiplication to two units.
        :param left_unit: The first unit to multiply.
        :param right_unit: The second unit to multiply.
        :return: A Unit representing the multiplication of the two.
        """
        # Convert to the correct units.
        left_unit = self.__left_unit_class(left_unit)
        right_unit = self.__right_unit_class(right_unit)

        # Initialize the multiplication.
        mul_unit = super().__call__(left_unit, right_unit)
        return cast(MulUnit, mul_unit)

    def __call__(self, value: UnitValue) -> MulUnit:
        """
        Creates a new compound unit of this type.
        :param value: The same value, in other units, or as a raw Numpy array.
        :return: The Unit object.
        """
        if isinstance(value, UnitInterface):
            if not self.is_compatible(value.type):
                # There's no reasonable way for us to convert a non-compound
                # unit to a compound one.
                raise UnitError("A compound unit with operation {} must be "
                                "initialized with another, not {}."
                                .format(self.operation,
                                        value.__class__.__name__))

            # Initialize using the left and right sub-unit values.
            return self.apply_to(value.left, value.right)

        else:
            # In this case, we'll just make one of the sub-units 1.
            left_unit = self.__left_unit_class(value)
            right_unit = self.__right_unit_class(1)

            mul_unit = super().__call__(left_unit, right_unit)
            return cast(MulUnit, mul_unit)

    def is_compatible(self, other: UnitType) -> bool:
        """
        See superclass for documentation.
        """
        if not isinstance(other, CompoundUnitType):
            # If it's not a compound unit, it's automatically not compatible.
            return False

        # We don't care which order the sub-units are in.
        sub_units_compatible = other.left.is_compatible(self.left) \
            and other.right.is_compatible(self.right)
        sub_units_compatible |= other.right.is_compatible(self.left) \
            and other.left.is_compatible(self.right)

        # Compound units are compatible if the compound unit operation is the
        # same, and the underlying sub-units have compatible types.
        return other.operation == self.operation and sub_units_compatible
