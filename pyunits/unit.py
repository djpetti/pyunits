import abc

import numpy as np

from .exceptions import UnitError
from .types import UnitValue
from .unit_base import UnitBase
from .unit_type import UnitType


class Unit(UnitBase, abc.ABC):
    """
    Base class for all units.
    """

    def __init__(self, unit_type: UnitType, value: UnitValue):
        """
        Initializes a new value of this unit.
        :param unit_type: The associated UnitType for this unit.
        :param value: The same value, in some other units, or as a raw numpy
        array.
        """
        super().__init__(unit_type)

        if isinstance(value, Unit):
            if not value.type.is_compatible(self.type):
                # We can't initialize a unit from the wrong type.
                raise UnitError("Cannot convert unit of type {} to unit"
                                " of type {}.".format(value.type_class,
                                                      self.type_class))

            # Initialize from the standard type.
            standard = value.to_standard()
            self._from_standard(standard)

        else:
            # We were passed a raw value.
            self._set_raw(np.asarray(value))

    def __str__(self) -> str:
        # Pretty-print the unit.
        return "{} {}".format(self.raw, self.name)

    def __eq__(self, other: UnitValue) -> bool:
        # Convert the other unit before comparing.
        this_class = self.type
        other_same = this_class(other)

        return np.array_equal(self.raw, other_same.raw)

    def _set_raw(self, raw: np.ndarray) -> None:
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

    @property
    def raw(self) -> np.ndarray:
        """
        See superclass for documentation.
        """
        return self.__value

    @property
    def name(self) -> str:
        """
        See superclass for documentation.
        """
        return self.__class__.__name__

    def cast_to(self, out_unit: UnitType) -> UnitBase:
        """
        See superclass for documentation.
        """
        out_type = out_unit.__class__
        return out_unit(self.type.as_type(self, out_type))
