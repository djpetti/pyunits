from typing import Type
import abc

import numpy as np

from . import types
from . import unit_type


class UnitInterface(abc.ABC):
    """
    Defines the public API that all units must implement.
    """

    @abc.abstractmethod
    def __neg__(self) -> 'UnitInterface':
        """
        Negates this unit.
        :return: The negation of the unit.
        """

    @abc.abstractmethod
    def __mul__(self, other: 'types.UnitValue') -> 'UnitInterface':
        """
        Multiplies this unit with another.
        :param other: The unit to multiply with.
        :return: The result of the multiplication.
        """

    @abc.abstractmethod
    def __rmul__(self, other: 'types.UnitValue') -> 'UnitInterface':
        """
        Implements reversed multiplication.
        :param other: The unit to multiply with.
        :return: The result of the multiplication.
        """

    @abc.abstractmethod
    def __truediv__(self, other: 'types.UnitValue') -> 'UnitInterface':
        """
        Divides this unit by another.
        :param other: The unit to divide by.
        :return: The result of the division.
        """

    @abc.abstractmethod
    def __rtruediv__(self, other: 'types.UnitValue') -> 'UnitInterface':
        """
        Implements reversed division.
        :param other: The unit to divide.
        :return: The result of the division.
        """

    @abc.abstractmethod
    def __add__(self, other: 'types.UnitValue') -> 'UnitInterface':
        """
        Implements normal addition.
        :param other: The unit to add.
        :return: The result of the addition.
        """

    @abc.abstractmethod
    def __radd__(self, other: 'types.UnitValue') -> 'UnitInterface':
        """
        Implements reversed addition.
        :param other: The unit to add.
        :return: The result of the addition.
        """

    @abc.abstractmethod
    def __sub__(self, other: 'types.UnitValue') -> 'UnitInterface':
        """
        Implements normal subtraction.
        :param other: The unit to subtract.
        :return: The result of the subtraction.
        """

    @abc.abstractmethod
    def __rsub__(self, other: 'types.UnitValue') -> 'UnitInterface':
        """
        Implements reversed subtraction.
        :param other: The unit to subtract.
        :return: The result of the subtraction.
        """

    @classmethod
    @abc.abstractmethod
    def is_standard(cls) -> bool:
        """
        :return: True if this is a standard unit, false otherwise.
        """

    @property
    @abc.abstractmethod
    def type(self) -> "unit_type.UnitType":
        """
        :return: The associated UnitType for this unit.
        """

    @property
    @abc.abstractmethod
    def type_class(self) -> Type:
        """
        :return: The class of the associated UnitType for this unit.
        """

    @abc.abstractmethod
    def to_standard(self) -> "UnitInterface":
        """
        Converts this unit to the standard unit for this unit type.
        :return: The same value in standard units.
        """

    @property
    @abc.abstractmethod
    def raw(self) -> np.ndarray:
        """
        :return: The raw value stored in this class.
        """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        :return: The name of the unit that will be used when printing.
        """

    @abc.abstractmethod
    def cast_to(self, out_type: "unit_type.UnitType") -> "UnitInterface":
        """
        Converts this unit to another unit of a different type.
        :param out_type: The UnitType that the output should be in the form
        of.
        :return: An instance of out_unit containing the converted value of this
        unit.
        """
