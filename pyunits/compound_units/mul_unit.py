import numpy as np

from ..types import UnitValue
from ..unit_base import UnitBase
from ..unit_interface import UnitInterface
from ..unit_type import UnitType
from .operations import Operation
from . import compound_unit_type


class MulUnit(UnitBase):
    """
    A pseudo-unit that is produced by the Mul unit-type to represent a compound
    unit comprised of the multiplication of two other units.
    """

    def __init__(self, unit_type: UnitType,
                 left_unit: UnitInterface, right_unit: UnitInterface):
        """
        :param unit_type: The associated UnitType for this unit.
        :param left_unit:  The first unit value to multiply.
        :param right_unit: The second unit value to multiply.
        """
        super().__init__(unit_type)

        self.__left_unit = left_unit
        self.__right_unit = right_unit

    def __mul__(self, other: UnitValue) -> "UnitInterface":
        if isinstance(other, UnitInterface) \
                and not other.type.is_compatible(self.type):
            # Create a (double) compound unit.
            mul_unit = self.type_class(Operation.MUL, self.type, other.type)
            return mul_unit.apply_to(self, other)

        return super().__mul__(other)

    def to_standard(self) -> UnitInterface:
        """
        See superclass for documentation.
        """
        # Convert both sub-units to standard form.
        standard_left = self.__left_unit.to_standard()
        standard_right = self.__right_unit.to_standard()

        # Create a new compound unit for the standard units.
        left_standard_type = standard_left.__class__
        right_standard_type = standard_right.__class__
        # type_class should always be CompoundUnitType.
        standard_mul = self.type_class(Operation.MUL,
                                       left_standard_type, right_standard_type)

        # Set the proper value for the standard compound unit.
        return standard_mul.apply_to(standard_left, standard_right)

    @property
    def raw(self) -> np.ndarray:
        """
        See superclass for documentation.
        """
        # Get the multiplied value.
        return self.__left_unit.raw * self.__right_unit.raw

    @property
    def name(self) -> str:
        """
        See superclass for documentation.
        """
        # Usually, the names of multiplied units are a concatenation of the
        # component units, for instance, Newton-meters is written Nm.
        return "{}{}".format(self.__left_unit.name, self.__right_unit.name)

    def cast_to(self, out_unit: "compound_unit_type.CompoundUnitType"
                ) -> "MulUnit":
        """
        See superclass for documentation.
        """
        # We'll cast each part of the compound unit individually, assuming we're
        # casting to another compound unit.
        left_out_class = out_unit.left
        right_out_class = out_unit.right

        left_casted = self.__left_unit.cast_to(left_out_class)
        right_casted = self.__right_unit.cast_to(right_out_class)

        # Create the correct output unit.
        return out_unit.apply_to(left_casted, right_casted)

    @property
    def left(self) -> UnitInterface:
        """
        :return: The unit that is the left-hand operand of the multiplication.
        """
        return self.__left_unit

    @property
    def right(self) -> UnitInterface:
        """
        :return: The unit that is the right-hand operand of the multiplication.
        """
        return self.__right_unit
