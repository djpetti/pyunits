from typing import Any, Callable
import functools
import inspect

from loguru import logger

import numpy as np

from .types import Numeric
from .unitless import Unitless


class WrapNumeric:
    """
    A decorator that handles automatically converting raw numeric types passed
    as arguments to Unitless instances. This allows us to simplify code that
    can accept raw numeric values, because we no longer have to do annoying
    type checks.

    For example, we can wrap a function like so:
    @WrapNumeric("foo")
    def my_awesome_function(foo: Unitless, bar: int) -> None:
        # Code here.

    Once we do that, all these calls will be valid:

    my_awesome_function(Unitless(5), 1)
    my_awesome_function(3.14, 2)
    my_awesome_function([1, 2, 3], 3)

    In the latter cases, the first argument will automatically be converted
    to a Unitless value inside the decorator.
    """

    def __init__(self, *args: str):
        """
        :param args: The names of all the arguments to the wrapped function
        that can be numeric types.
        """
        self.__arg_names = frozenset(args)
        self.__arg_positions = set()

    def __call__(self, to_wrap: Callable) -> Callable:
        """
        Wraps a function.
        :param to_wrap: The function to wrap.
        :return: The wrapped version of the function.
        """
        functools.update_wrapper(self, to_wrap)

        # Figure out which argument number corresponds to which named argument.
        signature = inspect.signature(to_wrap)
        for pos, name in enumerate(signature.parameters.keys()):
            if name in self.__arg_names:
                logger.debug("Parameter '{}' can be passed as argument {}.",
                             name, pos)
                self.__arg_positions.add(pos)

        @functools.singledispatch
        def convert_numeric(maybe_numeric: Any) -> Any:
            """
            If passed a raw numeric value, it creates a Unitless instance out
            of it and returns it. Otherwise, it is an identity.
            :param maybe_numeric: The value.
            :return: Either the same value, or a Unitless instance.
            """
            # By default, return the same value.
            return maybe_numeric

        @convert_numeric.register(np.ndarray)
        @convert_numeric.register(int)
        @convert_numeric.register(float)
        @convert_numeric.register(list)
        @convert_numeric.register(tuple)
        def _(maybe_numeric: Numeric) -> Unitless:
            return Unitless(maybe_numeric)

        def _wrap_numeric_impl(*args: Any, **kwargs: Any) -> Any:
            """
            The actual wrapper function.
            :param args: The positional arguments to pass to the wrapped
            function.
            :param kwargs: The keyword arguments to pass to the wrapped
            function.
            :return: The return value of the function.
            """
            # For any positional argument, we check if the argument at this
            # position was recorded as requiring conversion.
            wrapped_args = []
            for arg_pos, arg in enumerate(args):
                if arg_pos in self.__arg_positions:
                    # It needs to be converted.
                    wrapped_args.append(convert_numeric(arg))
                else:
                    wrapped_args.append(arg)

            # For any keyword arguments, we just look them up by name.
            wrapped_kwargs = {}
            for arg_name, value in kwargs.items():
                if arg_name in self.__arg_names:
                    # It needs to be converted.
                    wrapped_kwargs[arg_name] = convert_numeric(value)
                else:
                    wrapped_kwargs[arg_name] = value

            # Now we can safely call the function.
            return to_wrap(*wrapped_args, **wrapped_kwargs)

        return _wrap_numeric_impl
