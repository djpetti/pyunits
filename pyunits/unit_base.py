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

    def __mul__(self, other: UnitValue) -> "UnitInterface":
        # Convert the other unit before multiplying.
        this_class = self.type
        other_same = this_class(other)

        return this_class(self.raw * other_same.raw)

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
