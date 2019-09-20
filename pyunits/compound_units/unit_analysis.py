from typing import Any, cast, Dict, Iterable, List, Mapping, Tuple
import functools

from ..unit_type import UnitType
from .compound_unit_type import CompoundUnitType
from .operations import Operation


@functools.singledispatch
def _is_product(to_check: Any) -> bool:  # pragma: no cover
    """
    Checks whether something represents a product of two other units.
    :param to_check: The argument to check
    :return: True if it is a product, false otherwise.
    """
    # Not covered because this is essentially an assertion.
    raise NotImplementedError("_is_product() can only be used on UnitTypes.")


@_is_product.register
def _(to_check: UnitType) -> bool:
    # A normal UnitType does not represent a product.
    return False


@_is_product.register
def _(to_check: CompoundUnitType) -> bool:
    # A CompoundUnitType can be a product if it has the right operation.
    return to_check.operation == Operation.MUL


@functools.singledispatch
def _is_fraction(to_check: Any) -> bool:  # pragma: no cover
    """
    Checks whether something represents one unit divided by another.
    :param to_check: The argument to check.
    :return: True if it is a fraction, false otherwise.
    """
    # Not covered because this is essentially an assertion.
    raise NotImplementedError("_is_fraction() can only be used on UnitTypes.")


@_is_fraction.register
def _(to_check: UnitType) -> bool:
    # A normal UnitType does not represent a fraction.
    return False


@_is_fraction.register
def _(to_check: CompoundUnitType) -> bool:
    # A CompoundUnitType can be a fraction if it has the right operation.
    return to_check.operation == Operation.DIV


def flatten(to_flatten: UnitType) -> Tuple[Dict[UnitType, int],
                                           Dict[UnitType, int]]:
    """
    Decomposes a UnitType into a set of sub-types that make up the numerator and
    denominator. None of these sub-units will be compound.
    :param to_flatten: The Unit or UnitType to flatten.
    :return: The set of sub-units that make up the numerator and denominator,
    with the corresponding power of each sub-unit.
    """
    numerator = {}
    denominator = {}
    expandable_numerator = [to_flatten]
    expandable_denominator = []

    def flatten_compound(maybe_compound: UnitType, invert=False) -> bool:
        """
        Tries to flatten a UnitType.
        :param maybe_compound: The unit to expand, which might be a compound
        unit.
        :param invert: If true, it will be assumed that we are expanding
        something in the denominator, and therefore, should invert the logic
        for deciding which sub-units go in the numerator and which go in the
        denominator.
        :return: True if the unit was a compound unit and was expanded, false
        otherwise.
        """
        # We can cast presumptively because it doesn't actually perform a
        # runtime check.
        as_compound = cast(CompoundUnitType, maybe_compound)

        # Decide what goes on what side of the rational expression.
        same_side = expandable_numerator
        other_side = expandable_denominator
        if invert:
            same_side = expandable_denominator
            other_side = expandable_numerator

        if _is_product(maybe_compound):
            same_side.append(as_compound.left)
            same_side.append(as_compound.right)
            return True
        if _is_fraction(to_expand):
            same_side.append(as_compound.left)
            other_side.append(as_compound.right)
            return True

        return False

    while expandable_numerator or expandable_denominator:
        if expandable_numerator:
            to_expand = expandable_numerator.pop()

            if not flatten_compound(to_expand):
                # This unit is not compound and therefore cannot be flattened.
                if to_expand not in numerator:
                    numerator[to_expand] = 0
                numerator[to_expand] += 1

        if expandable_denominator:
            to_expand = expandable_denominator.pop()

            if not flatten_compound(to_expand, invert=True):
                # This unit is not compound and therefore cannot be flattened.
                if to_expand not in denominator:
                    denominator[to_expand] = 0
                denominator[to_expand] += 1

    return numerator, denominator


def un_flatten(numerator: Mapping[UnitType, int],
               denominator: Mapping[UnitType, int]) -> UnitType:
    """
    Converts flattened sets of numerator and denominator types to a single
    CompoundUnitType.
    :param numerator: The set of numerator types with corresponding powers.
    :param denominator: The set of denominator types with corresponding powers.
    :return: The CompoundUnitType it created.
    """
    def build_product(operands: Iterable[UnitType]) -> UnitType:
        """
        Builds a single CompoundUnitType from a set of types that we want to
        multiply together.
        :param operands: The types that we want to multiply.
        :return: The single CompoundUnitType it created.
        """
        # Python doesn't handle recursion well, so we do this using what I call
        # the "2048 algorithm".
        reduced = operands[:]
        while len(reduced) > 1:
            to_reduce = reduced
            reduced = []

            while to_reduce:
                next_reduced = to_reduce.pop()
                if to_reduce:
                    mul_with = to_reduce.pop()
                    next_reduced = CompoundUnitType(Operation.MUL, next_reduced,
                                                    mul_with)
                reduced.append(next_reduced)

        return reduced[0]

    def expand_powers(operands: Mapping[UnitType, int]) -> List[UnitType]:
        """
        Expands a set of UnitTypes from a dictionary of types and corresponding
        powers to a list of types where some may appear more than once.
        :param operands: The dictionary mapping UnitTypes to powers.
        :return: A list of the same UnitTypes.
        """
        expanded = []
        for unit_type, power in operands.items():
            expanded.extend([unit_type for _ in range(power)])

        return expanded

    # Convert the numerator and denominator into products.
    numerator = expand_powers(numerator)
    denominator = expand_powers(denominator)
    prod_numerator = build_product(numerator)
    if not denominator:
        # It's entirely valid not to have a denominator, in which case, we can
        # just return the numerator.
        return prod_numerator

    prod_denominator = build_product(denominator)
    # Perform the final division.
    return CompoundUnitType(Operation.DIV, prod_numerator, prod_denominator)


def simplify(to_simplify: CompoundUnitType) -> UnitType:
    """
    Takes an input, and puts it in the simplest possible form. For instance,
    we pass it CompoundUnitType representing (m * s) / s ^ 2, it would return a
    CompoundUnitType representing m / s.

    A note about conversions: This function is cautious, and will avoid making
    any changes that could change the numeric value of anything in the input
    unit. For instance, we should technically be able to simplify something
    like (N * m) / (in * s), because we have an implicit conversion from in to
    m. However, this would change the raw value of any instances of the input
    unit type. Therefore, we do not make this simplification automatically.
    :param to_simplify: The UnitType to simplify.
    :return: If no simplification can be performed, it simply returns the input.
    Otherwise, it returns a new, equivalent CompoundUnitType in the simplest
    form possible.
    """
    # Begin by flattening the input.
    numerator, denominator = flatten(to_simplify)

    # Since flattening removes any compound units, we can assume that types are
    # compatible iff a simple reference equality condition is satisfied.

    # Look for overlaps between the numerator and denominator.
    redundant_types = []
    for divisor in denominator:
        if divisor in numerator:
            # This is redundant, and we can remove it.
            redundant_types.append(divisor)

    if not redundant_types:
        # What we passed in can't be simplified.
        return to_simplify

    # Remove all the redundant stuff now.
    for unit_type in redundant_types:
        # Decrease the powers.
        denominator[unit_type] -= 1
        if denominator[unit_type] == 0:
            # Remove zero powers completely.
            denominator.pop(unit_type)

        numerator[unit_type] -= 1
        if numerator[unit_type] == 0:
            numerator.pop(unit_type)

    # Convert into a new CompoundUnitType.
    return un_flatten(numerator, denominator)
