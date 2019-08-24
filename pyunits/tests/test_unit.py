from typing import NamedTuple
import unittest.mock as mock

import numpy as np

import pytest

from pyunits.exceptions import UnitError
from pyunits.types import UnitValue
from pyunits import unit
from .helpers import MyUnit, MyStandardUnit


class TestUnit:
    """
    Tests for the Unit class.
    """

    class UnitConfig(NamedTuple):
        """
        Configuration for the tests.
        :param unit: The unit instance under test.
        :param standard_unit: The standard unit instance to test with.
        :param mock_type: The UnitType that the Unit was constructed with.
        :param mock_mul: The mock compound_units.Mul class.
        """
        unit: unit.Unit
        standard_unit: unit.Unit
        mock_type: mock.Mock
        mock_mul: mock.Mock

    @classmethod
    @pytest.fixture(params=[10, 5.0, np.array([1, 2, 3]), [1, 2, 3]])
    def config(cls, request) -> UnitConfig:
        """
        Creates a new configuration encapsulating a MyUnit object.
        :param request: The pytest object used for parametrization. Not
        type-annotated because the type is not easily accessible.
        :return: The Unit that it created.
        """
        # Fake the unit type so it's compatible.
        unit_type = mock.MagicMock()
        unit_type.is_compatible.return_value = True

        my_unit = MyUnit(unit_type, request.param)
        standard_unit = MyStandardUnit(unit_type, request.param)

        with mock.patch(unit.__name__ + ".Mul") as mock_mul:
            yield cls.UnitConfig(unit=my_unit, standard_unit=standard_unit,
                                 mock_type=unit_type, mock_mul=mock_mul)

            # Finalization done upon exit from context manager.

    @pytest.mark.parametrize("unit_value", [10, 5.0, np.array([1, 2, 3]),
                                            [1, 2, 3]])
    def test_init(self, unit_value: UnitValue) -> None:
        """
        Tests that we can initialize a unit properly.
        :param unit_value: The value to initialize the unit with, for testing.
        """
        # Arrange.
        # Create a fake type to use.
        unit_type = mock.Mock()
        unit_type.is_compatible.return_value = True

        # Convert whatever value we get to an array to use for assertions.
        expected_value = np.asarray(unit_value)

        # Act.
        # Create the unit.
        unit = MyStandardUnit(unit_type, unit_value)

        # Assert.
        # The raw value should be correct.
        np.testing.assert_array_equal(expected_value, unit.raw)

    def test_to_standard(self, config: UnitConfig) -> None:
        """
        Tests that to_standard works.
        :param config: The configuration to use for testing.
        """
        # Arrange done in fixtures.
        # Act.
        standard_converted = config.unit.to_standard()

        # It should have been correctly converted.
        assert isinstance(standard_converted, MyStandardUnit)
        np.testing.assert_array_equal(config.unit.raw *
                                      MyUnit.CONVERSION_FACTOR,
                                      standard_converted.raw)

    def test_to_standard_already_converted(self,
                                           config: UnitConfig) -> None:
        """
        Tests that to_standard works.
        :param config: The configuration to use for testing.
        """
        # Arrange done in fixtures.
        # Act.
        standard_not_converted = config.standard_unit.to_standard()

        # No conversion should have happened.
        assert isinstance(standard_not_converted, MyStandardUnit)
        np.testing.assert_array_equal(config.standard_unit.raw,
                                      standard_not_converted.raw)

    def test_init_from_other_unit(self, config: UnitConfig) -> None:
        """
        Tests that we can initialize one unit from another.
        :param config: The configuration to use for testing.
        """
        # Arrange done in fixtures.
        # Act.
        # Create a new unit from this one.
        new_unit = MyStandardUnit(config.mock_type, config.unit)

        # Assert.
        # It should have done the conversion.
        np.testing.assert_array_equal(config.unit.raw *
                                      MyUnit.CONVERSION_FACTOR,
                                      new_unit.raw)

    def test_init_wrong_type(self, config: UnitConfig) -> None:
        """
        Tests that it won't let us initialize a unit from another one if the
        types don't match.
        :param config: The configuration to use for testing.
        """
        # Arrange.
        # Make it look like the unit has an incompatible type.
        config.mock_type.is_compatible.return_value = False

        # Mock our own type so it also says it is incompatible. (The comparison
        # should be commutative.)
        my_type = mock.Mock()
        my_type.is_compatible.return_value = False

        # Act and assert.
        with pytest.raises(UnitError):
            MyStandardUnit(my_type, config.unit)

    def test_eq(self, config: UnitConfig) -> None:
        """
        Tests that two units will compare as equal when they should.
        :param config: The configuration to use for testing.
        """
        # Arrange.
        # It uses the type directly to convert the unit when comparing, so we're
        # going to have to mock the type a little bit more.
        mock_type = config.mock_type
        mock_type.side_effect = lambda other: MyUnit(mock_type, other)

        # Create a unit that is the exact same as this one.
        same_unit = MyUnit(mock_type, config.unit.raw)
        # Create a unit that is different.
        different_unit = MyUnit(mock_type, config.unit.raw + 1)

        # Act and assert.
        assert config.unit == same_unit
        assert config.unit != different_unit

    def test_eq_other_unit(self, config: UnitConfig) -> None:
        """
        Tests that two units will compare as equal when they are different
        units, but hold equivalent values.
        :param config: The configuration to use for testing.
        """
        # Arrange.
        # It uses the type directly to convert the unit when comparing, so we're
        # going to have to mock the type a little bit more.
        mock_type = config.mock_type
        mock_type.side_effect = lambda other: MyUnit(mock_type, other)

        # Create a unit that is equivalent to this one.
        same_unit = MyStandardUnit(mock_type,
                                   config.unit.raw * MyUnit.CONVERSION_FACTOR)
        # Create a unit that is not.
        different_unit = MyStandardUnit(mock_type, config.unit.raw)

        # Act and assert.
        assert config.unit == same_unit
        assert config.unit != different_unit

    def test_name(self, config: UnitConfig) -> None:
        """
        Tests that getting the name of a unit works.
        :param config: The configuration to use for testing.
        """
        # Arrange done in fixtures.
        # Act and assert.
        assert config.unit.name == MyUnit.__name__

    def test_str(self, config: UnitConfig) -> None:
        """
        Tests that converting a unit to a string works.
        :param config: The configuration to use for testing.
        """
        # Arrange done in fixtures.
        # Act.
        string_unit = str(config.unit)

        # Assert.
        assert string_unit == "{} {}".format(config.unit.raw, MyUnit.__name__)

    def test_cast_to(self, config: UnitConfig) -> None:
        """
        Tests that the casting helper works.
        :param config: The configuration to use for testing.
        """
        # Arrange.
        # Mock an output unit class.
        out_unit = mock.MagicMock()

        # Act.
        got_unit = config.unit.cast_to(out_unit)

        # Assert.
        out_type = out_unit.__class__
        # It should have performed the cast.
        config.mock_type.as_type.assert_called_once_with(config.unit, out_type)

        # It should have initialized the new unit instance.
        out_unit.assert_called_once_with(config.mock_type.as_type.return_value)
        assert got_unit == out_unit.return_value

    def test_mul_numeric(self, config: UnitConfig) -> None:
        """
        Tests that the multiplication operation works correctly when a unit is
        multiplied by a numeric value.
        :param config: The configuration to use.
        """
        # Arrange.
        expected_product = config.unit.raw * 5

        # Set up the mocks so that we can correctly make a new unit of the same
        # type.
        mock_unit_instance = config.mock_type.return_value
        mock_raw = mock.PropertyMock(return_value=expected_product)
        type(mock_unit_instance).raw = mock_raw

        # Act.
        # Multiply by a numeric value.
        product = config.unit * 5

        # Assert.
        # It should have created a new unit of the same type.
        assert config.mock_type.call_count == 1
        # It should have been created with the raw product value.
        my_args, _ = config.mock_type.call_args
        got_product = my_args[0]
        np.testing.assert_array_equal(expected_product, got_product)

        assert product == mock_unit_instance

    def test_mul_compatible_unit(self, config: UnitConfig) -> None:
        """
        Tests that the multiplication operation works correctly when a unit is
        multiplied by another with a compatible type.
        :param config: The configuration to use.
        """
        # Arrange.
        # Make sure converted units have the same type as their parent.
        converted = config.mock_type.return_value
        mock_type_property = mock.PropertyMock(return_value=config.mock_type)
        type(converted).type = mock_type_property

        # Act.
        product = config.unit * config.standard_unit

        # Assert.
        # It should have checked compatibility.
        config.mock_type.is_compatible.assert_called_once_with(config.mock_type)

        # It should have converted the standard unit.
        config.mock_type.assert_called_once_with(config.standard_unit)
        # It should have created a new compound unit.
        config.mock_mul.assert_called_once_with(config.mock_type,
                                                config.mock_type)

        compound_unit = config.mock_mul.return_value
        compound_unit.apply_to.assert_called_once_with(config.unit,
                                                       converted)

        # It should have returned the compound unit.
        assert product == compound_unit.apply_to.return_value

    def test_mul_incompatible_unit(self, config: UnitConfig) -> None:
        """
        Tests that the multiplication operation works correctly when a unit is
        multiplied by another with an incompatible type.
        :param config: The configuration to use.
        """
        # Arrange.
        # Create a fake type for the other unit.
        other_type = mock.Mock()
        other_type_property = mock.PropertyMock(return_value=other_type)
        type(config.standard_unit).type = other_type_property

        # Make it look like the types are incompatible.
        other_type.is_compatible.return_value = False

        # Act.
        product = config.unit * config.standard_unit

        # Assert.
        # It should have checked compatibility.
        other_type.is_compatible.assert_called_once_with(config.mock_type)
        # It should not have attempted to convert.
        config.mock_type.assert_not_called()

        # It should have created a new compound unit.
        config.mock_mul.assert_called_once_with(config.mock_type,
                                                other_type)

        compound_unit = config.mock_mul.return_value
        compound_unit.apply_to.assert_called_once_with(config.unit,
                                                       config.standard_unit)

        # It should have returned the compound unit.
        assert product == compound_unit.apply_to.return_value
