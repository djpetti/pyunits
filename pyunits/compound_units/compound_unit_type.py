from typing import cast, FrozenSet, Tuple, Union

from loguru import logger

from ..exceptions import UnitError
from ..types import UnitValue
from ..unit_interface import UnitInterface
from ..unit_type import UnitType
from .compound_unit import CompoundUnit
from .div_unit import DivUnit
from .mul_unit import MulUnit
from .operations import Operation


class CompoundUnitType(UnitType):
    """
    Unit type that represents the multiplication of two units.
    """

    # Maps Operations to the corresponding CompoundUnit subclasses.
    OPERATION_TO_CLASS = {Operation.MUL: MulUnit, Operation.DIV: DivUnit}

    def _init_new(self, operation: Operation,
                  left_unit_class: UnitType, right_unit_class: UnitType):
        """
        :param operation: The operation performed by the compound unit.
        :param left_unit_class: The class of the first unit to multiply.
        :param right_unit_class: The class of the second unit to multiply.
        """
        self.__enforce_compatibility_rules(operation, left_unit_class,
                                           right_unit_class)

        self.__operation = operation
        self.__left_unit_class = left_unit_class
        self.__right_unit_class = right_unit_class

        logger.debug("Creating new unit type {} with sub-units {} and {}.",
                     operation.name, left_unit_class.__class__.__name__,
                     right_unit_class.__class__.__name__)

        # Functionally, the class we're "wrapping" is CompoundUnit.
        super()._init_new(self.OPERATION_TO_CLASS[operation])

    @classmethod
    def _pre_hash(cls, operation: Operation, left_unit_class: UnitType,
                  right_unit_class: UnitType
                  ) -> Tuple[Operation, Union[FrozenSet[UnitType],
                                              Tuple[UnitType, UnitType]]]:
        """
        Transforms the arguments passed to get() before they are hashed, mainly
        so that equivalent product types hash to the same thing. See _init_new()
        for documentation on the parameters.
        :return: A tuple containing the arguments, with the left and right
        sub-types possibly in a set to indicate their lack of ordering.
        """
        sub_types = (left_unit_class, right_unit_class)
        if operation == Operation.MUL:
            # Multiplication is commutative, so express that by putting the
            # sub-types in a set.
            sub_types = frozenset(sub_types)

        return operation, sub_types

    @staticmethod
    def __enforce_compatibility_rules(operation: Operation,
                                      left_type: UnitType,
                                      right_type: UnitType) -> None:
        """
        Enforces rules about the compatibility of the two sub-units. These are
        mostly there to stop us from creating nonsensical units like in / m.
        :param operation: The operation to perform.
        :param left_type: The left subtype.
        :param right_type: The right subtype.
        """
        will_accept = True
        if left_type.is_compatible(right_type):
            if operation == Operation.DIV:
                # For division, we allow sub-unit compatibility under no
                # circumstances.
                will_accept = False

            elif left_type != right_type:
                # For multiplication, we only allow it if the two types are
                # identical, i.e. this is a squared unit.
                will_accept = False

        if not will_accept:
            raise UnitError("Sub-units {} and {} should not be compatible with"
                            " each-other.".format(left_type.__class__.__name__,
                                                  right_type.__class__.__name__)
                            )

    @property
    def left(self) -> UnitType:
        """
        :return: The first UnitType to multiply.
        """
        return self.__left_unit_class

    @property
    def right(self) -> UnitType:
        """
        :return: The second UnitType to multiply.
        """
        return self.__right_unit_class

    @property
    def operation(self) -> Operation:
        """
        :return: The operation being applied.
        """
        return self.__operation

    def apply_to(self, left_unit: UnitInterface,
                 right_unit: UnitInterface) -> CompoundUnit:
        """
        Applies the compound operation to two units.
        :param left_unit: The first unit to multiply.
        :param right_unit: The second unit to multiply.
        :return: A Unit representing the multiplication of the two.
        """
        # Convert to the correct units.
        left_unit = self.__left_unit_class(left_unit)
        right_unit = self.__right_unit_class(right_unit)

        # Initialize the multiplication.
        compound_unit = super().__call__(left_unit, right_unit)
        return cast(CompoundUnit, compound_unit)

    def __call__(self, value: UnitValue) -> CompoundUnit:
        """
        Creates a new compound unit of this type.
        :param value: The same value, in other units, or as a raw Numpy array.
        :return: The Unit object.
        """
        if isinstance(value, UnitInterface):
            if not self.is_compatible(value.type):
                # There's no reasonable way for us to convert a non-compound
                # unit to a compound one.
                raise UnitError("A compound unit with operation {} must be "
                                "initialized with another, not {}."
                                .format(self.operation,
                                        value.__class__.__name__))

            # Initialize using the left and right sub-unit values.
            value = cast(CompoundUnit, value)
            return self.apply_to(value.left, value.right)

        else:
            # In this case, we'll just make one of the sub-units 1.
            left_unit = self.__left_unit_class(value)
            right_unit = self.__right_unit_class(1)

            compound_unit = super().__call__(left_unit, right_unit)
            return cast(CompoundUnit, compound_unit)

    def standard_unit_class(self) -> 'CompoundUnitType':
        """
        See superclass for documentation.
        """
        # Find the standard unit classes for the sub-types.
        left_standard_class = self.left.standard_unit_class()
        right_standard_class = self.right.standard_unit_class()

        # Create a new compound type with these.
        return self.get(self.operation, left_standard_class,
                        right_standard_class)

    def is_compatible(self, other: UnitType) -> bool:
        """
        See superclass for documentation.
        """
        if not isinstance(other, CompoundUnitType):
            # If it's not a compound unit, it's automatically not compatible.
            return False

        sub_units_compatible = other.left.is_compatible(self.left) \
            and other.right.is_compatible(self.right)
        if self.operation == Operation.MUL:
            # Since multiplication is commutative, we don't care what order the
            # sub-units are in for this case.
            sub_units_compatible |= other.right.is_compatible(self.left) \
                and other.left.is_compatible(self.right)

        # Compound units are compatible if the compound unit operation is the
        # same, and the underlying sub-units have compatible types.
        return other.operation == self.operation and sub_units_compatible
