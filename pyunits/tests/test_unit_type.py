from typing import NamedTuple
import unittest.mock as mock

import pytest

from pyunits.unit import UnitError
from pyunits.unit_type import UnitType, CastHandler
from pyunits.exceptions import CastError
from .helpers import MyType, MyUnit, MyOtherType


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

    def test_as_type(self, wrapped_unit: MyType) -> None:
        """
        Tests that we can properly perform a cast using as_type().
        :param wrapped_unit: The decorated unit class.
        """
        # Arrange.
        # Create a new fake UnitType.
        other_type = mock.MagicMock()
        # Give it a name for easier debugging.
        other_type.__name__ = "OtherType"
        # A fake handler.
        handler = mock.Mock()

        # Register a cast for it.
        wrapped_unit.register_cast(other_type, handler)

        # A fake unit to convert.
        to_convert = MyUnit(10)

        # Act.
        converted = wrapped_unit.as_type(to_convert, other_type)

        # Assert.
        # It should have called the handler.
        handler.assert_called_once_with(to_convert)
        assert converted == handler.return_value

    def test_as_type_invalid_cast(self, wrapped_unit: MyType) -> None:
        """
        Tests that as_type() will refuse to perform an invalid cast.
        :param wrapped_unit: The decorated unit class.
        """
        # Arrange.
        # Create a new fake UnitType.
        other_type = mock.MagicMock()
        # Give it a name for easier debugging.
        other_type.__name__ = "OtherType"

        # A fake unit to convert.
        to_convert = MyUnit(10)

        # Act and assert.
        # Conversion should fail since no cast is registered.
        with pytest.raises(CastError):
            wrapped_unit.as_type(to_convert, other_type)

    class TestCastHandler:
        """
        Tests for the CastHandler class.
        """

        class HandlerConfig(NamedTuple):
            """
            Configuration for tests.
            :param handler: The wrapped handler function under test.
            :param mock_function: The mocked function that we are wrapping.
            :param wrapper_class: The wrapping instance of CastHandler.
            :param mock_from_unit: The mocked Unit class that we are converting
            from.
            :param mock_to_unit: The mocked Unit class that we are converting
            to.
            """
            handler: CastHandler.WrappedHandler
            mock_function: mock.Mock
            wrapper_class: CastHandler
            mock_from_unit: mock.Mock
            mock_to_unit: mock.Mock

        @classmethod
        @pytest.fixture
        def config(cls) -> HandlerConfig:
            """
            Generates configuration for tests,
            :return: The HandlerConfig that it created.
            """
            # Create the function to wrap.
            mock_function = mock.Mock()

            # Create fake unit classes.
            mock_from_unit = mock.MagicMock(spec=MyType)
            mock_from_unit.__name__ = "MockFromUnit"
            mock_to_unit = mock.MagicMock(spec=MyOtherType)
            mock_to_unit.__name__ = "MockToUnit"

            # Create the wrapper instance.
            wrapper_class = CastHandler(mock_from_unit, mock_to_unit)
            # Wrap the function.
            wrapped_handler = wrapper_class(mock_function)

            return cls.HandlerConfig(handler=wrapped_handler,
                                     mock_function=mock_function,
                                     wrapper_class=wrapper_class,
                                     mock_from_unit=mock_from_unit,
                                     mock_to_unit=mock_to_unit)

        def test_init(self, config: HandlerConfig) -> None:
            """
            Tests that initializing the wrapper works correctly.
            :param config: The HandlerConfig to use for this test.
            """
            # Arrange and act done by fixture.
            # Assert.
            # It should have registered the cast.
            config.mock_from_unit.register_cast.assert_called_once_with(
                config.mock_to_unit.__class__, mock.ANY
            )

        def test_init_same_type(self, config: HandlerConfig) -> None:
            """
            Tests that trying to cast to the input type causes an error.
            :param config: The HandlerConfig to use for this test.
            """
            # Arrange done in fixture.
            # Act and assert.
            # Creating the wrapper should fail when both unit are the same
            # type.
            with pytest.raises(CastError):
                CastHandler(config.mock_from_unit, config.mock_from_unit)

        def test_wrapped_call(self, config: HandlerConfig) -> None:
            """
            Tests that calling the wrapped handler function works as expected.
            :param config: The HandlerConfig to use for this test.
            """
            # Arrange.
            # Create a mock Unit to convert.
            mock_unit = mock.Mock(spec=MyUnit)

            # Act.
            # Try converting it.
            converted = config.handler(mock_unit)

            # Assert.
            # It should have made sure all the units were correct.
            config.mock_from_unit.assert_called_once_with(mock_unit)

            input_unit = config.mock_from_unit.return_value
            config.mock_function.assert_called_once_with(input_unit)

            raw_output = config.mock_function.return_value
            config.mock_to_unit.assert_called_once_with(raw_output)

            assert converted == config.mock_to_unit.return_value
