from typing import Type
import abc

import numpy as np

from . import unit_type


class UnitBase(abc.ABC):
    """
    Defines the public API that all units must implement.
    """

    def __init__(self, my_type: "unit_type.UnitType"):
        """
        :param my_type: The associated UnitType for this unit.
        """
        self.__type = my_type

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

    @abc.abstractmethod
    def to_standard(self) -> "UnitBase":
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
    def cast_to(self, out_unit: "unit_type.UnitType") -> "UnitBase":
        """
        Converts this unit to another unit of a different type.
        :param out_unit: The Unit class that the output should be in the form
        of.
        :return: An instance of out_unit containing the converted value of this
        unit.
        """
