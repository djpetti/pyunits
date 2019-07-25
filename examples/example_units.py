import numpy as np

from pyunits.unit import Unit
from pyunits.unit_type import UnitType


class Length(UnitType):
    """
    Type for length units.
    """
    pass


class Time(UnitType):
    """
    Type for time units.
    """
    pass


@Length
class Meters(Unit):
    """
    A meters unit.
    """

    def _from_raw(self, raw: np.ndarray) -> None:
        """
        See superclass for documentation.
        """
        self.__value = raw

    def _from_standard(self, standard_value: "Unit") -> None:
        """
        See superclass for documentation.
        """
        # This is the standard unit.
        self.__value = standard_value.raw

    def to_standard(self) -> Unit:
        """
        See superclass for documentation.
        """
        # This is the standard unit.
        return self

    @property
    def raw(self) -> np.ndarray:
        """
        See superclass for documentation.
        """
        return self.__value

    @property
    def name(self) -> str:
        """
        See superclass for documentation.
        """
        return "m"


@Length
class Centimeters(Unit):
    """
    A centimeters unit.
    """

    def _from_raw(self, raw: np.ndarray) -> None:
        """
        See superclass for documentation.
        """
        self.__value = raw

    def _from_standard(self, standard_value: "Unit") -> None:
        """
        See superclass for documentation.
        """
        # Convert from meters.
        self.__value = standard_value.raw * 100

    def to_standard(self) -> Unit:
        """
        See superclass for documentation.
        """
        # Convert to meters.
        return Meters(self.raw / 100)

    @property
    def raw(self) -> np.ndarray:
        """
        See superclass for documentation.
        """
        return self.__value

    @property
    def name(self) -> str:
        """
        See superclass for documentation.
        """
        return "cm"


@Time
class Seconds(Unit):
    """
    A seconds unit.
    """

    def _from_raw(self, raw: np.ndarray) -> None:
        """
        See superclass for documentation.
        """
        self.__value = raw

    def _from_standard(self, standard_value: "Unit") -> None:
        """
        See superclass for documentation.
        """
        # This is the standard unit.
        self.__value = standard_value.raw

    def to_standard(self) -> Unit:
        """
        See superclass for documentation.
        """
        # This is the standard unit.
        return self

    @property
    def raw(self) -> np.ndarray:
        """
        See superclass for documentation.
        """
        return self.__value

    @property
    def name(self) -> str:
        """
        See superclass for documentation.
        """
        return "s"
