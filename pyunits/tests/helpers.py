from pyunits.unit import StandardUnit, Unit
from pyunits.unit_type import UnitType


class MyType(UnitType):
    """
    A unit type that we can use for testing.
    """
    pass


class MyOtherType(UnitType):
    """
    Another unit type that we can use for testing.
    """
    pass


class MyStandardUnit(StandardUnit):
    """
    Represents the standard unit for this fake unit system. Deliberately
    un-decorated so that we can test decoration.
    """
    pass


class MyUnit(Unit):
    """
    A fake Unit subclass that we can use for testing. Deliberately undecorated
    so that we can test decoration.
    """

    # Conversion factor to use between this and the standard unit.
    CONVERSION_FACTOR = 50

    def _from_standard(self, standard_value: Unit) -> None:
        """
        See superclass for documentation.
        """
        # Use a fake conversion factor.
        self._set_raw(standard_value.raw / self.CONVERSION_FACTOR)

    def to_standard(self) -> Unit:
        """
        See superclass for documentation.
        :return:
        """
        # Normally, MyStandardUnit would be wrapped with a UnitType, so we
        # wouldn't have to manually pass the first parameter. However, for ease-
        # of-testing, it is not.
        return MyStandardUnit(self.type, self.raw * self.CONVERSION_FACTOR)
