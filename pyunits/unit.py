import abc

import numpy as np

from .compound_units import Div, Mul
from .exceptions import UnitError
from .arithmetic_helpers import do_mul, do_div, do_add
from .types import CompoundTypeFactories, UnitValue
from .unit_base import UnitBase
from .unit_interface import UnitInterface
from .unit_type import UnitType


class Unit(UnitBase, abc.ABC):
    """
    Base class for all units.
    """

    # The compound type factories that this class will use.
    COMPOUND_TYPE_FACTORIES = CompoundTypeFactories(mul=Mul, div=Div)

    def __init__(self, unit_type: UnitType, value: UnitValue):
        """
        Initializes a new value of this unit.
        :param unit_type: The associated UnitType for this unit.
        :param value: The same value, in some other units, or as a raw numpy
        array.
        """
        super().__init__(unit_type)

        if isinstance(value, UnitInterface):
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

    def __mul__(self, other: UnitValue) -> UnitInterface:
        return do_mul(self.COMPOUND_TYPE_FACTORIES, self, other)

    def __truediv__(self, other: UnitValue) -> UnitInterface:
        return do_div(self.COMPOUND_TYPE_FACTORIES, self, other)

    def __rtruediv__(self, other: UnitValue) -> UnitInterface:
        return do_div(self.COMPOUND_TYPE_FACTORIES, other, self)

    def __add__(self, other: UnitValue) -> UnitInterface:
        return do_add(self, other)

    def _set_raw(self, raw: np.ndarray) -> None:
        """
        Initializes this class with the given numeric value.
        :param raw: The raw value to use.
        """
        self.__value = raw

    @abc.abstractmethod
    def _from_standard(self, standard_value: 'StandardUnit') -> None:
        """
        Initializes this unit from a different unit with a "standard" value.
        :param standard_value: The standard unit to initialize from.
        """

    @classmethod
    def is_standard(cls) -> bool:
        """
        See superclass for documentation.
        """
        # We'll assume it's not standard.
        return False

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

    def cast_to(self, out_type: UnitType) -> UnitInterface:
        """
        See superclass for documentation.
        """
        out_type_class = out_type.__class__
        return out_type(self.type.as_type(self, out_type_class))


class StandardUnit(Unit):
    """
    Can be inherited from to identify that a particular unit is the "standard"
    unit for its UnitType. This is useful for two reasons: It makes standard
    units "explicit", so we can do nice things like raise an exception when we
    don't have one. Also, it saves us from having to write boilerplate code for
    standard units.
    """

    def _from_standard(self, standard_value: 'StandardUnit') -> None:
        """
        See superclass for documentation.
        """
        # This is the standard unit.
        self._set_raw(standard_value.raw)

    @classmethod
    def is_standard(cls) -> bool:
        """
        See superclass for documentation.
        """
        return True

    def to_standard(self) -> Unit:
        """
        See superclass for documentation.
        """
        # This is the standard unit.
        return self
