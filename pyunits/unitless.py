from typing import Union

import numpy as np

from .exceptions import UnitError
from .types import Numeric, UnitValue
from .unit_base import UnitBase
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
class Unitless(UnitBase):
    """
    Represents a unit-less value. By design, these are pretty locked-down, so
    that generally the user has to explicitly extract the raw value in order
    to do anything with them.
    """

    def __init__(self, unit_type: UnitlessType,
                 value: Union[Numeric, 'Unitless']):
        """
        :param unit_type: The UnitlessType instance used to create this class.
        :param value: The unitless value to wrap.
        """
        super().__init__(unit_type)

        if isinstance(value, UnitInterface):
            if not value.type.is_compatible(self.type):
                # Implicit initialization from normal units are disallowed,
                # because my judgement is that this has too high of a propensity
                # for causing bugs.
                raise UnitError("To initialize a Unitless value from another "
                                "unit, you must explicitly take the raw value.")

            self.__value = value.raw
        else:
            # Numeric type.
            self.__value = np.asarray(value)

    def __mul__(self, other: UnitValue) -> UnitInterface:
        """
        See superclass for documentation.
        """
        if not isinstance(other, UnitInterface):
            # Multiplying by a raw numeric value.
            # Our only concern in this case is making sure the result is a
            # Unitless value.
            return self.type(self.raw * other)
        elif isinstance(other, type(self)):
            # In this case, we just multiply the raw values.
            return self.type(self.raw * other.raw)
        else:
            # We don't handle normal multiplication in this class, and instead
            # rely on the other unit's reflected multiplication operation.
            return NotImplemented

    def __truediv__(self, other: UnitValue) -> UnitInterface:
        """
        See superclass for documentation.
        """
        if not isinstance(other, UnitInterface):
            # Dividing by a raw numeric value.
            return self.type(self.raw / other)
        if isinstance(other, type(self)):
            # In this case, we just divide the raw values.
            return self.type(self.raw / other.raw)
        else:
            # We don't handle normal division in this class, and instead rely
            # on the other unit's reflected division operator.
            return NotImplemented

    def __rtruediv__(self, other: UnitValue) -> UnitInterface:
        # The only way that we should ever get here is if we are trying to
        # divide a raw numerical value.
        assert not isinstance(other, UnitInterface)

        return self.type(other / self.raw)

    def __add__(self, other: UnitValue) -> UnitInterface:
        if not isinstance(other, UnitInterface):
            # Our only concern in this case is making sure the result is a
            # Unitless value.
            return self.type(self.raw + other)
        elif isinstance(other, type(self)):
            # In this case, we just add the raw values.
            return self.type(self.raw + other.raw)
        else:
            # We don't handle normal addition in this class, and instead
            # rely on the other unit's reflected addition operation.
            return NotImplemented

    @classmethod
    def is_standard(cls) -> bool:
        """
        See superclass for documentation.
        """
        # This unit is already in standard form.
        return True

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
        raise NotImplementedError("A unitless value should not ever need to be "
                                  "casted.")
