from typing import Type
import abc

import numpy as np

from .compound_units import compound_unit_type, unit_analysis
from .numeric_handling import WrapNumeric
from .types import UnitValue, CompoundTypeFactories
from .unitless import Unitless
from .unit_interface import UnitInterface
from .unit_type import UnitType


class UnitBase(UnitInterface, abc.ABC):
    """
    Base functionality for all Unit-like objects, including compound units.
    """

    def __init__(self, my_type: UnitType):
        """
        :param my_type: The associated UnitType for this unit.
        """
        self.__type = my_type

    def __str__(self) -> str:
        # Pretty-print the unit.
        return "{} {}".format(self.raw, self.name)

    def equals(self, other: UnitValue) -> bool:
        # Convert the other unit before comparing.
        this_class = self.type
        other_same = this_class(other)

        return np.array_equal(self.raw, other_same.raw)

    @WrapNumeric("other")
    def _do_mul(self, compound_type_factories: CompoundTypeFactories,
                other: UnitInterface) -> UnitInterface:
        """
        Helper that implements the multiplication operation.
        :param compound_type_factories: The factories to use for creating
        CompoundUnitTypes.
        :param other: The unit to multiply by this one.
        :return: The multiplication of the two units.
        """
        if other.type.is_compatible(self.type):
            # In this case, we'll get some unit squared. Convert to the
            # same units before proceeding.
            this_class = self.type
            other = this_class(other)

        # Create the compound unit.
        mul_unit_factory = compound_type_factories.mul(self.type,
                                                       other.type)
        mul_unit = mul_unit_factory.apply_to(self, other)
        return unit_analysis.simplify(mul_unit, compound_type_factories)

    @WrapNumeric("other")
    def _do_div(self, compound_type_factories: CompoundTypeFactories,
                other: UnitInterface) -> UnitInterface:
        """
        Helper that implements the division operation.
        :param compound_type_factories: The factories to use for creating
        CompoundUnitTypes.
        :param other: The unit to divide this one by.
        :return: The quotient of the two units. Note that this can be a unitless
        value if the inputs are of the same UnitType.
        """
        if other.type.is_compatible(self.type):
            # In this case, we'll get a unit-less value. Convert to the same
            # units before proceeding.
            this_class = self.type
            other = this_class(other)

            return Unitless(self.raw / other.raw)

        else:
            # Otherwise, create the compound unit.
            div_unit_factory = compound_type_factories.div(self.type,
                                                           other.type)
            div_unit = div_unit_factory.apply_to(self, other)
            return unit_analysis.simplify(div_unit, compound_type_factories)

    @property
    def type(self) -> UnitType:
        """
        :return: The associated UnitType for this unit.
        """
        return self.__type

    @property
    def type_class(self) -> Type:
        """
        :return: The class of the associated UnitType for this unit.
        """
        return self.__type.__class__
