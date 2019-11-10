from typing import cast
import abc
import functools

from ..arithmetic_helpers import do_mul, do_div, do_add
from ..types import CompoundTypeFactories, UnitValue
from ..unit_base import UnitBase
from ..unit_interface import UnitInterface
from .operations import Operation
from .unit_analysis import simplify
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

    def __get_type_factories(self) -> CompoundTypeFactories:
        """
        Helper function that creates the proper CompoundTypeFactories
        record for this class.
        :return: The CompoundTypeFactories that it created.
        """
        mul_type = functools.partial(self.type.get, Operation.MUL)
        div_type = functools.partial(self.type.get, Operation.DIV)
        return CompoundTypeFactories(mul=mul_type, div=div_type)

    def __mul__(self, other: UnitValue) -> UnitInterface:
        return do_mul(self.__get_type_factories(), self, other)

    def __truediv__(self, other: UnitValue) -> UnitInterface:
        return do_div(self.__get_type_factories(), self, other)

    def __rtruediv__(self, other: UnitValue) -> UnitInterface:
        return do_div(self.__get_type_factories(), other, self)

    def __add__(self, other: UnitValue) -> UnitInterface:
        return do_add(self, other)

    def is_standard(self) -> bool:
        """
        See superclass for documentation.
        """
        # This unit will be considered standard if both its subunits are.
        return self.left.is_standard() and self.right.is_standard()

    def to_standard(self) -> UnitInterface:
        """
        See superclass for documentation.
        """
        # Convert both sub-units to standard form.
        standard_left = self.__left_unit.to_standard()
        standard_right = self.__right_unit.to_standard()

        # Create a new compound unit with the standard unit values.
        standard_compound_type = self.type.standard_unit_class()
        standard_compound_type = cast('compound_unit_type.CompoundUnitType',
                                      standard_compound_type)
        standard = standard_compound_type.apply_to(standard_left,
                                                   standard_right)

        # This might call for simplification.
        return simplify(standard, self.__get_type_factories())

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

    @property
    def operation(self) -> Operation:
        """
        :return: The operation performed by this unit.
        """
        my_type = cast(compound_unit_type.CompoundUnitType, self.type)
        return my_type.operation
