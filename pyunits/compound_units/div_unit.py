import numpy as np

from .compound_unit import CompoundUnit


class DivUnit(CompoundUnit):
    """
    A pseudo-unit that is produced by a CompoundUnitType to represent a compound
    unit comprised of the division of two other units.
    """

    @property
    def raw(self) -> np.ndarray:
        """
        See superclass for documentation.
        """
        return self.left.raw / self.right.raw
