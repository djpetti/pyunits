from .compound_units import unit_analysis
from .numeric_handling import WrapNumeric
from .types import CompoundTypeFactories
from .unit_interface import UnitInterface
from .unitless import Unitless


@WrapNumeric("left", "right")
def do_mul(compound_type_factories: CompoundTypeFactories,
           left: UnitInterface, right: UnitInterface) -> UnitInterface:
    """
    Helper that implements the multiplication operation.
    :param compound_type_factories: The factories to use for creating
    CompoundUnitTypes.
    :param left: The left-hand unit to multiply.
    :param right: The right-hand unit to multiply.
    :return: The multiplication of the two units.
    """
    if right.type.is_compatible(left.type):
        # In this case, we'll get some unit squared. Convert to the
        # same units before proceeding.
        left_class = left.type
        right = left_class(right)

    # Create the compound unit.
    mul_unit_factory = compound_type_factories.mul(left.type,
                                                   right.type)
    mul_unit = mul_unit_factory.apply_to(left, right)
    return unit_analysis.simplify(mul_unit, compound_type_factories)


@WrapNumeric("left", "right")
def do_div(compound_type_factories: CompoundTypeFactories,
           left: UnitInterface, right: UnitInterface) -> UnitInterface:
    """
    Helper that implements the division operation.
    :param compound_type_factories: The factories to use for creating
    CompoundUnitTypes.
    :param left: The left-hand unit to multiply.
    :param right: The right-hand unit to multiply.
    :return: The quotient of the two units. Note that this can be a unitless
    value if the inputs are of the same UnitType.
    """
    if right.type.is_compatible(left.type):
        # In this case, we'll get a unit-less value. Convert to the same
        # units before proceeding.
        left_type = left.type
        right = left_type(right)

        return Unitless(left.raw / right.raw)

    else:
        # Otherwise, create the compound unit.
        div_unit_factory = compound_type_factories.div(left.type,
                                                       right.type)
        div_unit = div_unit_factory.apply_to(left, right)
        return unit_analysis.simplify(div_unit, compound_type_factories)
