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
        return super().__call__(left_unit, right_unit)

    def __call__(self, value: UnitValue) -> MulUnit:
        """
        Creates a new compound unit of this type.
        :param value: The same value, in other units, or as a raw Numpy array.
        :return: The Unit object.
        """
        if isinstance(value, UnitInterface):
            if not isinstance(value, MulUnit):
                # There's no reasonable way for us to convert a non-compound
                # unit to a compound one.
                raise UnitError("A multiplied compound unit must be initialized"
                                " with another, not {}."
                                .format(value.__class__.__name__))

            # Initialize using the left and right sub-unit values.
            return self.apply_to(value.left, value.right)

        else:
            # In this case, we'll just make one of the sub-units 1.
            left_unit = self.__left_unit_class(value)
            right_unit = self.__right_unit_class(1)

            return super().__call__(left_unit, right_unit)

    def is_compatible(self, other: "CompoundUnitType") -> bool:
        """
        See superclass for documentation.
        """
        # We don't care which order the sub-units are in.
        sub_units_compatible = other.left.is_compatible(self.left) \
            and other.right.is_compatible(self.right)
        sub_units_compatible |= other.right.is_compatible(self.left) \
            and other.left.is_compatible(self.right)

        # Compound units are compatible if the compound unit type is the same,
        # and the underlying sub-units have compatible types.
        return other.__class__ == self.__class__ and sub_units_compatible
