from typing import NamedTuple
import unittest.mock as mock

import pytest

from pyunits.exceptions import UnitError
from pyunits.unit_type import UnitType, CastHandler
from pyunits.exceptions import CastError
from .helpers import MyType, MyStandardUnit, MyUnit, MyOtherType


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
        MyType.clear_interning_cache()
        return MyType.decorate(MyUnit)

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
        assert my_unit.type == wrapped_unit

    def test_wrapping_standard(self) -> None:
        """
        Tests that wrapping a standard unit works under normal conditions.
        """
        # Arrange.
        MyType.clear_interning_cache()
        # Make it look like we have no standard unit for this type.
        MyType._STANDARD_UNIT_CLASS = None

        # Act.
        # Perform the decoration.
        wrapped = MyType.decorate(MyStandardUnit)

        # Assert.
        # It should have set the standard unit.
        assert MyType.standard_unit_class() == wrapped

    def test_wrapping_standard_twice(self) -> None:
        """
        Tests that it fails if we try to specify multiple standard units for a
        single UnitType.
        """
        # Arrange.
        MyType.clear_interning_cache()
        # Make it look like we have no standard unit for this type.
        MyType._STANDARD_UNIT_CLASS = None

        # Specify a standard unit.
        MyType.decorate(MyStandardUnit)

        # Act and assert.
        # Trying to specify another one should raise an exception.
        with pytest.raises(UnitError, match="already has standard"):
            MyType.decorate(MyStandardUnit)

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
        to_convert = MyUnit(wrapped_unit, 10)

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
        to_convert = MyUnit(wrapped_unit, 10)

        # Act and assert.
        # Conversion should fail since no cast is registered.
        with pytest.raises(CastError):
            wrapped_unit.as_type(to_convert, other_type)

    def test_standard_unit_class_not_set(self) -> None:
        """
        Tests that getting the standard unit class fails when none was set.
        """
        # Arrange.
        MyType.clear_interning_cache()
        # Make it look like we have no standard unit for this type.
        MyType._STANDARD_UNIT_CLASS = None

        # Act and assert.
        with pytest.raises(UnitError, match="no standard"):
            MyType.standard_unit_class()

    def test_is_compatible(self, wrapped_unit: MyType) -> None:
        """
        Tests that is_compatible works under normal conditions.
        :param wrapped_unit: The decorated Unit class.
        """
        # Arrange.
        fake_unit_class = mock.Mock(spec=MyUnit)
        fake_unit_class.is_standard.return_value = False

        # Create an instance of another UnitType.
        other_type = MyOtherType.decorate(fake_unit_class)

        # Create another instance of the same unit type.
        same_type = MyType.decorate(fake_unit_class)

        # Act.
        compatible_with_other = wrapped_unit.is_compatible(other_type)
        compatible_with_same = wrapped_unit.is_compatible(same_type)

        # Assert.
        # It should not be compatible with the other type.
        assert not compatible_with_other
        # It should be compatible with the same type.
        assert compatible_with_same

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
            MyType.clear_interning_cache()
            MyOtherType.clear_interning_cache()

            # Create the function to wrap.
            mock_function = mock.Mock()

            # Create fake unit classes.
            mock_from_unit = mock.MagicMock(spec=MyType.decorate(MyUnit))
            mock_from_unit.__name__ = "MockFromUnit"
            mock_to_unit = mock.MagicMock(spec=MyOtherType.decorate(MyUnit))
            mock_to_unit.__name__ = "MockToUnit"

            # Make it look like the two units are not compatible.
            mock_to_unit.is_compatible.return_value = False
            mock_from_unit.is_compatible.return_value = False

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
            # Arrange.
            # Make it look like both units are the same type.
            config.mock_from_unit.is_compatible.return_value = True
            config.mock_to_unit.is_compatible.return_value = True

            # Act and assert.
            # Creating the wrapper should fail when both units are the same
            # type.
            with pytest.raises(CastError):
                CastHandler(config.mock_from_unit, config.mock_to_unit)

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
