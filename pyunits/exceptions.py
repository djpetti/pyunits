class UnitError(Exception):
    """
    General error for units.
    """
    pass


class CastError(Exception):
    """
    Raised when a cast fails.
    """


class InterningError(Exception):
    """
    Raised when we try to use an interned class in a way that would break the
    pattern.
    """
