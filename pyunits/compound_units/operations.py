import enum


class Operation(enum.IntEnum):
    """
    Kinds of compound units we can have.
    """
    # A multiplication compound unit.
    MUL = enum.auto()
    # A division compound unit.
    DIV = enum.auto()
