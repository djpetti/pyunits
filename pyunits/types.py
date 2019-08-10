from typing import Iterable, Union

import numpy as np


# Type alias for what we accept when initializing units.
UnitValue = Union["Unit", np.ndarray, int, float, Iterable]
