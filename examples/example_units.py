from pyunits.unit import StandardUnit, Unit
from pyunits.unit_type import UnitType, CastHandler


import numpy as np


class Length(UnitType):
    """
    Type for length units.
    """


class Length2D(UnitType):
    """
    Type for 2D length units.
    """


class Time(UnitType):
    """
    Type for time units.
    """


class Energy(UnitType):
    """
    Type for energy units.
    """


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

    def _from_standard(self, standard_value: StandardUnit) -> None:
        """
        See superclass for documentation.
        """
        # Convert from meters.
        self._set_raw(standard_value.raw * 100)

    def to_standard(self) -> Meters:
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


@Length.decorate
class Kilometers(Unit):
    """
    A kilometers unit.
    """

    def _from_standard(self, standard_value: StandardUnit) -> None:
        """
        See superclass for documentation.
        """
        # Convert from meters.
        self._set_raw(standard_value.raw / 1000)

    def to_standard(self) -> Meters:
        """
        See superclass for documentation.
        """
        # Convert to meters.
        return Meters(self.raw * 1000)

    @property
    def name(self) -> str:
        """
        See superclass for documentation.
        """
        return "km"


@Length.decorate
class Inches(Unit):
    """
    An inches unit.
    """

    def _from_standard(self, standard_value: StandardUnit) -> None:
        """
        See superclass for documentation.
        """
        # Convert from meters.
        self._set_raw(standard_value.raw / 0.0254)

    def to_standard(self) -> Meters:
        """
        See superclass for documentation.
        """
        # Convert to meters.
        return Meters(self.raw * 0.0254)

    @property
    def name(self) -> str:
        """
        See superclass for documentation.
        """
        return "in"


@Length.decorate
class Miles(Unit):
    """
    A miles unit.
    """

    def _from_standard(self, standard_value: StandardUnit) -> None:
        """
        See superclass for documentation.
        """
        # Convert from meters.
        self._set_raw(standard_value.raw * 0.000621371)

    def to_standard(self) -> Meters:
        """
        See superclass for documentation.
        """
        # Convert to meters.
        return Meters(self.raw / 0.000621371)

    @property
    def name(self) -> str:
        """
        See superclass for documentation.
        """
        return "mi"


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


@Time.decorate
class Minutes(Unit):
    """
    A minutes unit.
    """

    def _from_standard(self, standard_value: StandardUnit) -> None:
        """
        See superclass for documentation.
        """
        # Convert from seconds.
        self._set_raw(standard_value.raw / 60)

    def to_standard(self) -> Seconds:
        """
        See superclass for documentation.
        """
        # Convert to seconds.
        return Seconds(self.raw * 60)

    @property
    def name(self) -> str:
        """
        See superclass for documentation.
        """
        return "min"


@Time.decorate
class Years(Unit):
    """
    A years unit.
    """

    def _from_standard(self, standard_value: StandardUnit) -> None:
        """
        See superclass for documentation.
        """
        # Convert from seconds.
        self._set_raw(standard_value.raw / 31536000)

    def to_standard(self) -> Seconds:
        """
        See superclass for documentation.
        """
        # Convert to seconds.
        return Seconds(self.raw * 31536000)

    @property
    def name(self) -> str:
        """
        See superclass for documentation.
        """
        return "yr"


@Energy.decorate
class Joules(StandardUnit):
    """
    A Joules unit.
    """

    @property
    def name(self) -> str:
        """
        See superclass for documentation.
        """
        return "J"


@CastHandler(Meters, Meters2D)
def meters_to_meters2d(meters: Meters) -> np.ndarray:
    """
    Cast for meters to 2D meters. Will make the second dimension 0.
    :param meters: The input, as meters.
    :return: The raw value to use for the Meters2D instance.
    """
    return np.append(meters.raw, 0)
