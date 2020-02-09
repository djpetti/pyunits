from typing import Dict, Mapping

from ..unit_interface import UnitInterface
from .unit_analysis import flatten


def _group_by_power(name_to_power: Mapping[str, int]) -> Dict[int, str]:
    """
    Groups unit names by the powers that they are raised to.
    For instance, if the input is {"m": 2, "s": 2, "k": 1}, it would output
    {2: "m*s", 1: "k"}.
    :param name_to_power: Maps the name of each unit to its power.
    :return: A dictionary mapping a power to a string representation of the
    products of the units that are raised to this power.
    """
    # Map powers to lists of unit names.
    powers_to_names = {}
    for name, power in name_to_power.items():
        if power not in powers_to_names:
            powers_to_names[power] = []
        powers_to_names[power].append(name)

    # Create combined string representations for each.
    for power, names in powers_to_names.items():
        powers_to_names[power] = "*".join(names)

    return powers_to_names


def _align_center(to_align: str, length: int) -> str:
    """
    Aligns a string so that it has a certain length and is center-justified.
    :param to_align: The string to align.
    :param length: The target length of the string. We will pad the string
    on either side with spaces to reach this length.
    :return: The padded string.
    """
    assert length >= len(to_align), "Length cannot be shorter than string."

    per_side_padding_length = (length - len(to_align)) // 2
    padding = " " * per_side_padding_length
    padded = f"{padding}{to_align}{padding}"

    # Because of integer division, we could have an off-by-one error.
    if len(padded) != length:
        padded += " "

    return padded


def _pretty_product(name_to_power: Mapping[str, int]) -> str:
    """
    Creates a pretty-printed representation of a unit product.
    :param name_to_power: Maps the name of each unit to its power.
    :return: A pretty-printed string representing the unit product.
    """
    # Group by power first.
    powers_to_names = _group_by_power(name_to_power)

    # If we only have one group, we can print it straight. Otherwise, we need to
    # parenthesize each individual group.
    parenthesize = len(powers_to_names) > 1
    product_str = ""
    for power, group in powers_to_names.items():
        # Raise to the appropriate power.
        if power != 1:
            group = f"{group}^{power}"
        if parenthesize:
            # Wrap in parentheses.
            group = f"({group})"

        product_str += group

    return product_str


def _pretty_radical(numerator: str, denominator: str) -> str:
    """
    Pretty-prints a radical.
    :param numerator: The pretty-printed numerator.
    :param denominator: The pretty-printed denominator.
    :return: The complete pretty-printed radical.
    """
    if denominator == "":
        # If we have no denominator, we simply print the numerator.
        return numerator
    if numerator == "":
        # An empty numerator can happen if we have, for instance, a single
        # unitless value. In this case, we just print a "1".
        numerator = "1"

    # Figure out how long the horizontal line should be.
    line_length = max(len(numerator), len(denominator)) + 2
    # Create the line.
    separator = "-" * line_length

    # Center the numerator and denominator.
    numerator = _align_center(numerator, line_length)
    denominator = _align_center(denominator, line_length)

    return f"{numerator}\n{separator}\n{denominator}"


def pretty_name(unit: UnitInterface) -> str:
    """
    Creates a pretty representation of a unit name. This will handle
    compound units correctly. For instance, if you give it an instance of the
    compound unit DivUnit(MulUnit(Meters, Meters), Second), it would produce

      m^2
     -----
       s

    :param unit: The unit to produce a pretty name for.
    :return: The pretty-printed name of the unit.
    """
    # Use the flattened representation, which is easier to work with.
    numerator, denominator = flatten(unit)

    # Extract the names of all the sub-units.
    numerator_names = {u.name: p for u, p in numerator.items()}
    denominator_names = {u.name: p for u, p in denominator.items()}

    # Pretty-print the radical.
    pretty_numerator = _pretty_product(numerator_names)
    pretty_denominator = _pretty_product(denominator_names)
    return _pretty_radical(pretty_numerator, pretty_denominator)
