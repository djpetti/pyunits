import pytest

from pyunits.unit import UnitError
from pyunits.unit_type import UnitType
from .helpers import MyType, MyUnit


class _OtherType(UnitType):
    """
    A dummy unit type to use for testing.
    """
    pass


class TestUnitType:
    """
    Tests for the unit_type class.
    """

    @classmethod
    @pytest.fixture
    def wrapped_unit(cls) -> MyType:
        """
        Wraps a unit class in our custom type decorator, to simulate the
        decorator being applied.
        :return: The wrapped unit class.
        """
        # Clear any previous types set on the unit.
        MyUnit.UNIT_TYPE = None

        return MyType(MyUnit)

    def test_wrapping(self, wrapped_unit: MyType) -> None:
        """
        Tests that it properly decorates a unit class.
        :param wrapped_unit: The decorated unit class.
        """
        # Arrange done in fixture.
        # Act.
        my_unit = wrapped_unit(10)

        # Assert.
        # The unit type should be correct.
        assert my_unit.UNIT_TYPE == MyType

    def test_wrapping_already_registered(self, wrapped_unit: MyType) -> None:
        """
        Tests that it refuses to change the type of a unit when it is already
        set.
        :param wrapped_unit: The decorated unit class.
        """
        # Arrange.
        # Set the type to something incompatible.
        MyUnit.UNIT_TYPE = _OtherType

        # Act and assert.
        with pytest.raises(UnitError):
            wrapped_unit(10)

    def test_double_wrap(self, wrapped_unit: MyType) -> None:
        """
        Tests that it properly handles the case where the class has already been
        decorated with this type.
        :param wrapped_unit: The decorated unit class.
        """
        # Arrange.
        # Set the type to the same thing.
        MyUnit.UNIT_TYPE = MyType

        # Act.
        my_unit = wrapped_unit(10)

        # Assert.
        # The unit type should still be correct.
        assert my_unit.UNIT_TYPE == MyType
