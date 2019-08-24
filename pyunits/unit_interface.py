from typing import Type
import abc

import numpy as np

from . import unit_type


class UnitInterface(abc.ABC):
    """
    Defines the public API that all units must implement.
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
