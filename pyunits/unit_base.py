from typing import Type
import abc

import numpy as np

from . import unit_type
from .types import UnitValue
from .unit_interface import UnitInterface


class UnitBase(UnitInterface, abc.ABC):
    """
    Base functionality for all Unit-like objects, including compound units.
    """

    def __init__(self, my_type: "unit_type.UnitType"):
        """
        :param my_type: The associated UnitType for this unit.
        """
        self.__type = my_type

    def __str__(self) -> str:
        # Pretty-print the unit.
        return "{} {}".format(self.raw, self.name)

    def __eq__(self, other: UnitValue) -> bool:
        # Convert the other unit before comparing.
        this_class = self.type
        other_same = this_class(other)

        return np.array_equal(self.raw, other_same.raw)

    def _do_mul(self, mul_type: Type, other: UnitValue) -> UnitInterface:
        """
        Helper that implements the multiplication operation.
        :param mul_type: The UnitType class to use for the compound Mul unit.
        :param other: The unit to multiply by this one.
        :return: The multiplication of the two units.
        """
        if isinstance(other, UnitInterface):
            if other.type.is_compatible(self.type):
                # In this case, we'll get some unit squared. Convert to the
                # same units before proceeding.
                this_class = self.type
                other = this_class(other)

            # Create the compound unit.
            mul_unit = mul_type(self.type, other.type)
            return mul_unit.apply_to(self, other)

        else:
            # A normal numeric value can be directly multiplied.
            return self.type(self.raw * other)

    def _do_div(self, div_type: Type, other: UnitValue) -> UnitValue:
        """
        Helper that implements the division operation.
        :param div_type: The UnitType class to use for the compound Div unit.
        :param other: The unit to divide this one by.
        :return: The quotient of the two units. Note that this can be a unitless
        value if the inputs are of the same UnitType.
        """
        if isinstance(other, UnitInterface):
            if other.type.is_compatible(self.type):
                # In this case, we'll get a unit-less value. Convert to the same
                # units before proceeding.
                this_class = self.type
                other = this_class(other)

                return self.raw / other.raw

            else:
                # Otherwise, create the compound unit.
                div_unit = div_type(self.type, other.type)
                return div_unit.apply_to(self, other)

        else:
            # A normal numeric value can be directly divided.
            return self.type(self.raw / other)

    @property
    def type(self) -> "unit_type.UnitType":
        """
        :return: The associated UnitType for this unit.
        """
        return self.__type

    @property
    def type_class(self) -> Type:
        """
        :return: The class of the associated UnitType for this unit.
        """
        return self.__type.__class__
