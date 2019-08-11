from functools import partial

from .operations import Operation
from .compound_unit_type import CompoundUnitType


# Shortcuts for creating compound units.
Mul = partial(CompoundUnitType, Operation.MUL)
Div = partial(CompoundUnitType, Operation.DIV)