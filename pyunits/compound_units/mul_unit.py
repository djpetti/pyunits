import numpy as np

from .compound_unit import CompoundUnit


class MulUnit(CompoundUnit):
    """
    A pseudo-unit that is produced by a CompoundUnitType to represent a compound
    unit comprised of the multiplication of two other units.
    """

    @property
    def raw(self) -> np.ndarray:
        """
        See superclass for documentation.
        """
        # Get the multiplied value.
        return self.left.raw * self.right.raw

    @property
    def name(self) -> str:
        """
        See superclass for documentation.
        """
        # Usually, the names of multiplied units are a concatenation of the
        # component units, for instance, Newton-meters is written Nm.
        return "{}{}".format(self.left.name, self.right.name)
