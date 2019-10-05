from typing import Type

import numpy as np

from .types import Numeric, UnitValue
from .unit_interface import UnitInterface
from .unit_type import UnitType


class UnitlessType(UnitType):
    """
    A UnitType that represents a unitless value. This may seem kind of silly,
    but it makes things easier in some regards, because it allows us to
    do things like guarantee that a division operation always returns a unit
    and not occasionally a raw value, and to represent constructs like
    s ^ -1.

    We make it so that UnitlessTypes are not compatible with anything. You
    might think it more logical to instead have them be compatible with
    everything, but this introduces a nasty reflexivity problem into
    compatibility checks, e.g. the result is different depending on
    whether you do a.is_compatible(b) or b.is_compatible(a). Furthermore,
    I think it's better for PyUnits to force users to initialize a new
    unit if they want to give a unit-less value units, instead of just
    doing it implicitly.
    """


@UnitlessType.decorate
class Unitless(UnitInterface):
    """
    Represents a unit-less value. By design, these are pretty locked-down, so
    that generally the user has to explicitly extract the raw value in order
    to do anything with them.
    """

    def __init__(self, unit_type: UnitlessType, value: Numeric):
        """
        :param unit_type: The UnitlessType instance used to create this class.
        :param value: The unitless value to wrap.
        """
        self.__type = unit_type
        self.__value = np.asarray(value)

    def __mul__(self, other: UnitValue) -> UnitInterface:
        """
        See superclass for documentation.
        """
        if isinstance(other, UnitInterface):
            # In this case, we just multiply the raw values.
            return other.type(other.raw * self.raw)
        else:
            # If it's a raw numeric value, the only consideration we need to
            # make is keeping it wrapped in a Unitless value.
            return self.type(other * self.raw)

    def __truediv__(self, other: UnitValue) -> UnitInterface:
        """
        See superclass for documentation.
        """
        if isinstance(other, Unitless):
            # In this case, we just divide the raw values.
            return other.type(self.raw / other.raw)
        elif isinstance(other, UnitInterface):
            # We don't handle normal division in this class, and instead rely
            # on the other unit's reflected division operator.
            raise NotImplementedError("Division of a unitless value is not "
                                      "implemented.")
        else:
            # If it's a raw numeric value, the only consideration we need to
            # make is keeping it wrapped in a Unitless value.
            return self.type(self.raw / other)

    @property
    def type(self) -> UnitlessType:
        """
        See superclass for documentation.
        """
        return self.__type

    @property
    def type_class(self) -> Type:
        """
        See superclass for documentation.
        """
        return self.type.__class__

    def to_standard(self) -> 'Unitless':
        """
        See superclass for documentation.
        """
        # The standard version of a unitless value is just itself.
        return self

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
        # The name is always empty.
        return ""

    def cast_to(self, out_type: UnitType) -> UnitInterface:
        """
        See superclass for documentation.
        """
        # If you're trying to cast a unitless value, something has gone
        # terribly wrong.
        raise ValueError("A unitless value should not ever need to be casted.")
