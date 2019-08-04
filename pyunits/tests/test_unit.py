import unittest.mock as mock

import numpy as np

import pytest

from pyunits.unit import Unit, UnitError
from .helpers import MyUnit, MyStandardUnit


class TestUnit:
    """
    Tests for the Unit class.
    """

    @classmethod
    @pytest.fixture(params=[10, 5.0, np.array([1, 2, 3]), [1, 2, 3]])
    def my_unit(cls, request) -> MyUnit:
        """
        Creates a new _MyUnit object.
        :param request: The pytest object used for parametrization. Not
        type-annotated because the type is not easily accessible.
        :return: The Unit that it created.
        """
        # Fake the unit type so it's compatible.
        MyUnit.UNIT_TYPE = "MyUnitType"

        return MyUnit(request.param)

    @classmethod
    @pytest.fixture(params=[10, 5.0, np.array([1, 2, 3]), [1, 2, 3]])
    def my_standard_unit(cls, request) -> MyStandardUnit:
        """
        Creates a new _MyStandardUnit object.
        :param request: The pytest object used for parametrization. Not
        type-annotated because the type is not easily accessible.
        :return: The Unit that it created.
        """
        # Fake the unit type so it's compatible.
        MyStandardUnit.UNIT_TYPE = "MyUnitType"

        return MyStandardUnit(request.param)

    @pytest.mark.parametrize("unit_value", [10, 5.0, np.array([1, 2, 3]),
                                            [1, 2, 3]])
    def test_init(self, unit_value: Unit.UnitValue) -> None:
        """
        Tests that we can initialize a unit properly.
        :param unit_value: The value to initialize the unit with, for testing.
        """
        # Arrange.
        # Convert whatever value we get to an array to use for assertions.
        expected_value = np.asarray(unit_value)

        # Act.
        # Create the unit.
        unit = MyStandardUnit(unit_value)

        # Assert.
        # The raw value should be correct.
        np.testing.assert_array_equal(expected_value, unit.raw)

    def test_to_standard(self, my_unit: MyUnit) -> None:
        """
        Tests that to_standard works.
        :param my_unit: The unit instance to test with.
        """
        # Arrange done in fixtures.
        # Act.
        standard_converted = my_unit.to_standard()

        # It should have been correctly converted.
        assert isinstance(standard_converted, MyStandardUnit)
        np.testing.assert_array_equal(my_unit.raw * MyUnit.CONVERSION_FACTOR,
                                      standard_converted.raw)

    def test_to_standard_already_converted(self, my_standard_unit:
                                           MyStandardUnit) -> None:
        """
        Tests that to_standard works.
        :param my_standard_unit: The standard unit instance to test with.
        """
        # Arrange done in fixtures.
        # Act.
        standard_not_converted = my_standard_unit.to_standard()

        # No conversion should have happened.
        assert isinstance(standard_not_converted, MyStandardUnit)
        np.testing.assert_array_equal(my_standard_unit.raw,
                                      standard_not_converted.raw)

    def test_init_from_other_unit(self, my_unit: MyUnit) -> None:
        """
        Tests that we can initialize one unit from another.
        :param my_unit: The unit instance to test with.
        """
        # Arrange done in fixtures.
        # Act.
        # Create a new unit from this one.
        new_unit = MyStandardUnit(my_unit)

        # Assert.
        # It should have done the conversion.
        np.testing.assert_array_equal(my_unit.raw * MyUnit.CONVERSION_FACTOR,
                                      new_unit.raw)

    def test_init_wrong_type(self, my_unit: MyUnit) -> None:
        """
        Tests that it won't let us initialize a unit from another one if the
        types don't match.
        :param my_unit: The unit instance to test with.
        """
        # Arrange.
        # Make it look like the unit has an incompatible type.
        MyUnit.UNIT_TYPE = "MyNewUnitType"

        # Act and assert.
        with pytest.raises(UnitError):
            MyStandardUnit(my_unit)

    def test_eq(self, my_unit: MyUnit) -> None:
        """
        Tests that two units will compare as equal when they should.
        :param my_unit: The unit instance to test with.
        """
        # Arrange.
        # Create a unit that is the exact same as this one.
        same_unit = MyUnit(my_unit.raw)
        # Create a unit that is different.
        different_unit = MyUnit(my_unit.raw + 1)

        # Act and assert.
        assert my_unit == same_unit
        assert my_unit != different_unit

    def test_eq_other_unit(self, my_unit: MyUnit) -> None:
        """
        Tests that two units will compare as equal when they are different
        units, but hold equivalent values.
        :param my_unit: The unit instance to test with.
        """
        # Arrange.
        # Create a unit that is equivalent to this one.
        same_unit = MyStandardUnit(my_unit.raw * MyUnit.CONVERSION_FACTOR)
        # Create a unit that is not.
        different_unit = MyStandardUnit(my_unit.raw)

        # Act and assert.
        assert my_unit == same_unit
        assert my_unit != different_unit

    def test_name(self, my_unit: MyUnit) -> None:
        """
        Tests that getting the name of a unit works.
        :param my_unit: The unit instance to test with.
        """
        # Arrange done in fixtures.
        # Act and assert.
        assert my_unit.name == MyUnit.__name__

    def test_str(self, my_unit: MyUnit) -> None:
        """
        Tests that converting a unit to a string works.
        :param my_unit: The unit instance to test with.
        """
        # Arrange done in fixtures.
        # Act.
        string_unit = str(my_unit)

        # Assert.
        assert string_unit == "{} {}".format(my_unit.raw, MyUnit.__name__)

    def test_cast_to(self, my_unit: MyUnit) -> None:
        """
        Tests that the casting helper works.
        :param my_unit: The unit instance to test with.
        """
        # Arrange.
        # Mock an output unit class.
        out_unit = mock.MagicMock()
        # Also mock the unit type.
        my_unit.UNIT_TYPE = mock.Mock()

        # Act.
        got_unit = my_unit.cast_to(out_unit)

        # Assert.
        out_type = out_unit.__class__
        # It should have performed the cast.
        my_unit.UNIT_TYPE.cast_to.assert_called_once_with(my_unit, out_type)

        # It should have initialized the new unit instance.
        out_unit.assert_called_once_with(my_unit.UNIT_TYPE.cast_to.return_value)
        assert got_unit == out_unit.return_value
