from typing import Union
import abc

import numpy as np

from .exceptions import UnitError


class Unit(abc.ABC):
    """
    Base class for all units.
    """

    # Type of unit this is.
    UNIT_TYPE = None

    def __init__(self, value: Union["Unit", np.ndarray]):
        """
        Initializes a new value of this unit.
        :param value: The same value, in some other units.
        """
        if isinstance(value, Unit):
            if value.UNIT_TYPE != self.UNIT_TYPE:
                # We can't initialize a unit from the wrong type.
                raise UnitError("Cannot convert unit of type {} to unit"
                                " of type {}.".format(value.UNIT_TYPE,
                                                      self.UNIT_TYPE))

            # Initialize from the standard type.
            standard = value.to_standard()
            self._from_standard(standard)

        else:
            # We were passed a raw value.
            self._from_raw(np.asarray(value))

    def __str__(self) -> str:
        # Pretty-print the unit.
        return "{} {}".format(self.raw, self.name)

    def _from_raw(self, raw: np.ndarray) -> None:
        """
        Initializes this class with the given numeric value.
        :param raw: The raw value to use.
        """
        self.__value = raw

    @abc.abstractmethod
    def _from_standard(self, standard_value: "Unit") -> None:
        """
        Initializes this unit from a different unit with a "standard" value.
        :param standard_value: The standard unit to initialize from.
        """
        pass

    @abc.abstractmethod
    def to_standard(self) -> "Unit":
        """
        Converts this unit to the standard unit for this unit type.
        :return: The same value in standard units.
        """
        pass

    @property
    def raw(self) -> np.ndarray:
        """
        :return: The raw value stored in this class.
        """
        return self.__value

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        :return: The name of the unit that will be used when printing.
        """
        pass
