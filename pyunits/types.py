from typing import Any, Iterable, NamedTuple, Type, Union

import numpy as np


# Type alias for numeric values that can be used to initialize a unit.
Numeric = Union[np.ndarray, int, float, Iterable]
# Type alias for what we accept when initializing units.
UnitValue = Union['unit_interface.UnitInterface', Numeric]

# The type of the Pytest request object. This is not easily accessible, so for
# now we just set it to Any.
RequestType = Any


class CompoundTypeFactories(NamedTuple):
    """
    Enumerates the CompoundUnitType classes we want to use for creating compound
    units.
    :param mul_type: The multiplication type.
    :param div_type: The division type.
    """
    mul: Type
    div: Type
