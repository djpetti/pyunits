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

    @property
    def name(self) -> str:
        """
        See superclass for documentation.
        """
        # Usually, the names of divided units are written as a fraction, so
        # we'll try to pretty-print it that way.
        numerator = self.left.name
        denominator = self.right.name

        separator_length = max(len(numerator), len(denominator))
        separator = "-" * separator_length

        return "{}\n{}\n{}".format(numerator, separator, denominator)
