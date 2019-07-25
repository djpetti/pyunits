from typing import Callable, NamedTuple, Type
import abc
import functools

from loguru import logger

from .exceptions import UnitError
from .unit import Unit


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

    # Type representing the handler function for a cast.
    CastHandler = Callable[[Unit], Unit]

    # This is a table that tells us what casts we can perform directly. It is
    # indexed by Casts, and the values are functions that perform that cast.
    _DIRECT_CASTS = {}

    @classmethod
    def register_cast(cls, cast: Cast, handler: CastHandler) -> None:
        """
        Registers a new cast that can be performed.
        :param cast: The cast to register.
        :param handler: The function that will perform this cast.
        """
        # Add the cast.
        logger.debug("Registering cast: {}", cast)
        cls._DIRECT_CASTS[cast] = handler

    def __init__(self, unit_class: Type):
        """
        :param unit_class: Allows UnitType classes to be used as class
        decorators for units. This is how we define the type of a unit.
        """
        functools.update_wrapper(self, unit_class)

        self.__unit_class = unit_class

    def __call__(self, *args, **kwargs) -> Type:
        """
        "Stamps" the unit class so we know what type it is.
        :param args: Will be forwarded to the Unit constructor.
        :param kwargs: Will be forwarded to the Unit constructor.
        :return: The Unit object.
        """
        cls = self.__class__

        # Stamp the unit class.
        if self.__unit_class.UNIT_TYPE is not None and \
                self.__unit_class.UNIT_TYPE != cls:
            # It already has a type.
            raise UnitError("Unit {} already has type {}, cannot make it type"
                            " {}.".format(self.__unit_class.__name__,
                                          self.__unit_class.UNIT_TYPE.__name__,
                                          cls.__name__))

        if self.__unit_class.UNIT_TYPE != cls:
            logger.debug("Registering unit {} as type {}.",
                         self.__unit_class.__name__, cls.__name__)
            self.__unit_class.UNIT_TYPE = cls

        return self.__unit_class(*args, **kwargs)



