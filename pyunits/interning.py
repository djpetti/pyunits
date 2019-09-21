from typing import Any, Callable, Tuple
import abc

from loguru import logger

from .exceptions import InterningError


class Interned(abc.ABC):
    """
    Implements the interning pattern. Instances of the same class that are
    constructed with the same arguments will be regarded as equivalent.

    Note that all the ctor arguments to the wrapped class must be hashable.
    """

    # Type to use for pre-hashing functions.
    _PreHashType = Callable[[Any, Any], Tuple]

    # Maps tuples of the class and the ctor kwargs to canonical instances.
    _INSTANCES = {}

    def __init__(self, *args, _expect_creation: bool = False, **kwargs):
        """
        All parameters are ignored except for _expect_creation.
        :param _expect_creation: Basically, if not set to true, this method will
        throw an exception. It is done like this to guard against the user
        accidentally trying to create an interned class normally.
        """
        if not _expect_creation:
            raise InterningError("Please use get() to create a new instance of "
                                 "this class.")

    @classmethod
    def _pre_hash(cls, *args: Any, **kwargs: Any) -> Tuple:
        """
        Specifies a custom transformation to run on the arguments passed to
        get() before hashing. It will be forwarded these arguments directly, and
        return a tuple of some sort. This can be useful to, for example, ignore
        certain arguments.
        :param args: Positional arguments passed to get().
        :param kwargs: Keyword arguments passed to get().
        :return: A tuple of transformed arguments.
        """
        return tuple(args) + tuple(kwargs.values())

    @abc.abstractmethod
    def _init_new(self, *args: Any, **kwargs: Any) -> None:
        """
        Called when we want to initialize a new instance of the derived class.
        It will be forwarded any arguments passed to get().
        :param args: Positional arguments passed to get().
        :param kwargs: Keyword arguments passed to get().
        """

    @classmethod
    def get(cls, *args: Any, **kwargs: Any) -> 'Interned':
        """
        Gets an instance of the derived class, returning a cached one if deemed
        appropriate.
        :param args: Will be forwarded to _init_new.
        :param kwargs: Will be forwarded to _init_new.
        :return: The instance that it created.
        """
        # Transform the arguments before hashing.
        arg_signature = cls._pre_hash(*args, **kwargs)
        # Add the class to create the full signature.
        signature = (cls, arg_signature)

        # See if this exists.
        instance = cls._INSTANCES.get(signature)
        if instance is None:
            # We have to create this instance.
            logger.debug("Creating new canonical instance: {}", signature)
            instance = cls(_expect_creation=True)
            instance._init_new(*args, **kwargs)

            # Save it for later.
            cls._INSTANCES[signature] = instance

        return instance

    @classmethod
    def clear_interning_cache(cls) -> None:
        """
        Forcefully clears any cached instances. This is mostly useful for
        testing, where we want to force it to create a new instance every time.
        """
        cls._INSTANCES = {}
