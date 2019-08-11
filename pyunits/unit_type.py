from typing import Callable, NamedTuple, Type
import abc
import functools

from loguru import logger

from .exceptions import CastError
from .types import UnitValue
from . import unit_interface

# Type alias for the function that does the casting.
CastFunction = Callable[["unit_interface.UnitInterface"],
                        "unit_interface.UnitInterface"]


class UnitType(abc.ABC):
    """
    Represents a type of unit.

    Two units are of the same type if we can convert one to the other and back
    again without losing information. For example, we might have a unit type
    "Length", and units of this type could be "Meters", "Inches", etc.

    We might be able to convert one unit type to another. This is called
    casting, and has the potential to lose information.
    """

    class Cast(NamedTuple):
        """
        Represents a cast.
        :param from_type: The type we want to cast from.
        :param to_type: The type we want to cast to.
        """
        from_type: Type
        to_type: Type

    # This is a table that tells us what casts we can perform directly. It is
    # indexed by Casts, and the values are functions that perform that cast.
    _DIRECT_CASTS = {}

    def __init__(self, unit_class: Type):
        """
        :param unit_class: Allows UnitBaseType classes to be used as class
        decorators for units. This is how we define the type of a unit.
        """
        functools.update_wrapper(self, unit_class)

        self.__unit_class = unit_class

    def __call__(self, *args, **kwargs) -> "unit_interface.UnitInterface":
        """
        "Stamps" the unit class so we know what type it is.
        :param args: Will be forwarded to the UnitBase constructor.
        :param kwargs: Will be forwarded to the UnitBase constructor.
        :return: The UnitBase object.
        """
        return self.__unit_class(self, *args, **kwargs)

    @classmethod
    def register_cast(cls, out_type: Type, handler: CastFunction) -> None:
        """
        Registers a new cast that can be performed.
        :param out_type: The UnitBaseType that we want to be able to convert this
        one too.
        :param handler: The function that will perform this cast.
        """
        cast = cls.Cast(from_type=cls, to_type=out_type)

        # Add the cast.
        logger.debug("Registering cast: {}", cast)
        cls._DIRECT_CASTS[cast] = handler

    @classmethod
    def as_type(cls, unit: "unit_interface.UnitInterface",
                out_type: Type) -> "unit_interface.UnitInterface":
        """
        Casts the wrapped unit to a new type.
        :param unit: The unit instance to convert.
        :param out_type: The unit type to cast to.
        :return: An equivalent unit of the specified type.
        """
        # Get the source unit type.
        from_type = cls
        logger.debug("Trying to cast from {} to {}.", from_type.__name__,
                     out_type.__name__)

        # Find the handler for this cast.
        cast = cls.Cast(from_type=from_type, to_type=out_type)
        if cast not in cls._DIRECT_CASTS:
            # We don't have a handler for it.
            raise CastError("Cannot cast from {} to {}."
                            .format(from_type.__name__, out_type.__name__))

        handler = cls._DIRECT_CASTS[cast]
        return handler(unit)

    def is_compatible(self, other: "UnitType") -> bool:
        """
        Checks if this type is equivalent to another for the purposes of
        conversion.
        :param other: The other type.
        :return: True if the two are equivalent, false otherwise.
        """
        # By default, types are equivalent if they are of the same class.
        return self.__class__ == other.__class__


class CastHandler:
    """
    Decorator for handling unit type casts. It can be used as follows:

    @CastHandler(FirstUnitBase, SecondUnit)
    def handle_cast(unit: FirstUnitBase) -> np.ndarray:
        # Do the conversion and return the value that will be passed to
        # SecondUnitBase's ctor.
    """

    # Type alias for the wrapped handler function.
    WrappedHandler = Callable[["unit_interface.UnitInterface"], UnitValue]

    def __init__(self, from_unit: UnitType, to_unit: UnitType):
        """
        :param from_unit: The unit that this handler will take as input.
        :param to_unit: The unit that this handler will produce as output.
        """

        if from_unit.is_compatible(to_unit):
            # We don't need a cast for this.
            raise CastError("Units {} and {} are both of type {} and are thus"
                            " directly convertible."
                            .format(from_unit.__name__, to_unit.__name__,
                                    from_unit.__class__.__name__))

        self.__from_unit = from_unit
        self.__to_unit = to_unit

    def __call__(self, func: WrappedHandler) -> CastFunction:
        """
        Wraps the function.
        :param func: The function being wrapped.
        :return: The wrapped function.
        """
        functools.update_wrapper(self, func)

        def wrapped(to_convert: "unit_interface.UnitInterface"
                    ) -> "unit_interface.UnitInterface":
            """
            Wrapper implementation.
            Does the conversion, ensuring that the input and output are in the
            correct units.
            :param to_convert: The UnitBase instance to convert.
            :return: The converted unit instance.
            """
            # Make sure the input is in the expected units.
            to_convert = self.__from_unit(to_convert)
            # Call the handler.
            raw_output = func(to_convert)
            # Make sure the output is in the expected units.
            return self.__to_unit(raw_output)

        # Register this cast.
        self.__from_unit.register_cast(self.__to_unit.__class__, wrapped)

        return wrapped
