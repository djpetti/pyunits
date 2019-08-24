from typing import Any, Iterable, Union

import numpy as np

from . import unit_interface


# Type alias for numeric values that can be used to initialize a unit.
Numeric = Union[np.ndarray, int, float, Iterable]
# Type alias for what we accept when initializing units.
UnitValue = Union["unit_interface.UnitInterface", Numeric]

# The type of the Pytest request object. This is not easily accessible, so for
# now we just set it to Any.
RequestType = Any
