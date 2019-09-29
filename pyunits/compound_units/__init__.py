from functools import partial

from .operations import Operation
from .compound_unit_type import CompoundUnitType


# Shortcuts for creating compound units.
Mul = partial(CompoundUnitType.get, Operation.MUL)
Div = partial(CompoundUnitType.get, Operation.DIV)
