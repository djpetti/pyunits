from typing import cast
import abc

from ..unit_base import UnitBase
from ..unit_interface import UnitInterface
from . import compound_unit_type


class CompoundUnit(UnitBase, abc.ABC):
    """
    A base class for compound units.
    """

    def __init__(self, unit_type: "compound_unit_type.CompoundUnitType",
                 left_unit: UnitInterface, right_unit: UnitInterface):
        """
        :param unit_type: The associated UnitType for this unit.
        :param left_unit:  The first unit value to multiply.
        :param right_unit: The second unit value to multiply.
        """
        super().__init__(unit_type)

        self.__left_unit = left_unit
        self.__right_unit = right_unit

    def to_standard(self) -> UnitInterface:
        """
        See superclass for documentation.
        """
        # Convert both sub-units to standard form.
        standard_left = self.__left_unit.to_standard()
        standard_right = self.__right_unit.to_standard()

        # Create a new compound unit with the standard unit values.
        compound_type = cast(compound_unit_type.CompoundUnitType, self.type)
        return compound_type.apply_to(standard_left, standard_right)

    def cast_to(self, out_type: "compound_unit_type.CompoundUnitType"
                ) -> "CompoundUnit":
        """
        See superclass for documentation.
        """
        # We'll cast each part of the compound unit individually, assuming we're
        # casting to another compound unit.
        left_out_class = out_type.left
        right_out_class = out_type.right

        left_casted = self.__left_unit.cast_to(left_out_class)
        right_casted = self.__right_unit.cast_to(right_out_class)

        # Create the correct output unit.
        return out_type.apply_to(left_casted, right_casted)

    @property
    def left(self) -> UnitInterface:
        """
        :return: The unit that is the left-hand operand.
        """
        return self.__left_unit

    @property
    def right(self) -> UnitInterface:
        """
        :return: The unit that is the right-hand operand.
        """
        return self.__right_unit
