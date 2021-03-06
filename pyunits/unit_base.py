from typing import Type
import abc

import numpy as np

from .types import UnitValue
from .unit_interface import UnitInterface
from .unit_type import UnitType


class UnitBase(UnitInterface, abc.ABC):
    """
    Base functionality for all Unit-like objects, including compound units.
    """

    def __init__(self, my_type: UnitType):
        """
        :param my_type: The associated UnitType for this unit.
        """
        self.__type = my_type

    def __str__(self) -> str:
        # Pretty-print the unit. If the unit representation is more than one
        # line, we want the number to show up in the middle vertically.
        name_lines = self.name.split("\n")
        middle_line_index = len(name_lines) // 2

        number_str = str(self.raw)
        middle_line = name_lines[middle_line_index]
        middle_with_number = "{} {}".format(number_str, middle_line)

        # Pad the other lines so that the unit representation is aligned
        # horizontally.
        padding = " " * (len(number_str) + 1)
        name_lines = [padding + line for line in name_lines]

        name_lines[middle_line_index] = middle_with_number

        return "\n".join(name_lines)

    def __neg__(self) -> UnitInterface:
        return self.type(-self.raw)

    def __radd__(self, other: UnitValue) -> UnitInterface:
        # It never matters which way we do addition, so just use the normal
        # operator.
        return self.__add__(other)

    def __sub__(self, other: UnitValue) -> UnitInterface:
        # Subtraction is just adding a negative value.
        return self.__add__(-other)

    def __rsub__(self, other: UnitValue) -> UnitInterface:
        # Reversed subtraction can also be implemented using addition.
        negated = -self
        return negated.__add__(other)

    def __rmul__(self, other: UnitValue) -> UnitInterface:
        # It never matters which way we do multiplication, so just use the
        # normal operator.
        return self.__mul__(other)

    def equals(self, other: UnitValue) -> bool:
        # Convert the other unit before comparing.
        this_class = self.type
        other_same = this_class(other)

        return np.array_equal(self.raw, other_same.raw)

    @property
    def type(self) -> UnitType:
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
