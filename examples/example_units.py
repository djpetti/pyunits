from pyunits.unit import StandardUnit, Unit
from pyunits.unit_type import UnitType, CastHandler


import numpy as np


class Length(UnitType):
    """
    Type for length units.
    """
    pass


class Length2D(UnitType):
    """
    Type for 2D length units.
    """
    pass


class Time(UnitType):
    """
    Type for time units.
    """
    pass


@Length.decorate
class Meters(StandardUnit):
    """
    A meters unit.
    """

    @property
    def name(self) -> str:
        """
        See superclass for documentation.
        """
        return "m"


@Length.decorate
class Centimeters(Unit):
    """
    A centimeters unit.
    """

    def _from_standard(self, standard_value: "Unit") -> None:
        """
        See superclass for documentation.
        """
        # Convert from meters.
        self._set_raw(standard_value.raw * 100)

    def to_standard(self) -> Unit:
        """
        See superclass for documentation.
        """
        # Convert to meters.
        return Meters(self.raw / 100)

    @property
    def name(self) -> str:
        """
        See superclass for documentation.
        """
        return "cm"


@Time.decorate
class Seconds(StandardUnit):
    """
    A seconds unit.
    """

    @property
    def name(self) -> str:
        """
        See superclass for documentation.
        """
        return "s"


@Length2D.decorate
class Meters2D(StandardUnit):
    """
    A unit for a 2D position in meters.
    """

    @property
    def name(self) -> str:
        """
        See superclass for documentation.
        """
        return "m"


@CastHandler(Meters, Meters2D)
def meters_to_meters2d(meters: Meters) -> np.ndarray:
    """
    Cast for meters to 2D meters. Will make the second dimension 0.
    :param meters: The input, as meters.
    :return: The raw value to use for the Meters2D instance.
    """
    return np.append(meters.raw, 0)
