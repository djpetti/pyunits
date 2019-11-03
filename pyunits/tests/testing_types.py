from typing import Callable
import unittest.mock as mock


# Type alias for a function that makes Units.
UnitFactory = Callable[..., mock.Mock]
# Type alias for a function that makes UnitTypes.
UnitTypeFactory = Callable[[str], mock.Mock]
