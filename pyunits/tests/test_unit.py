import numpy as np

import pytest

from pyunits.unit import Unit, UnitError


class _MyStandardUnit(Unit):
    """
    Represents the standard unit for this fake unit system.
    """

    def _from_standard(self, standard_value: Unit) -> None:
        """
        See superclass for documentation.
        """
        # It is already in standard form.
        self._set_raw(standard_value.raw)

    def to_standard(self) -> Unit:
        """
        See superclass for documentation.
        """
        return self


class _MyUnit(Unit):
    """
    A fake Unit subclass that we can use for testing.
    """

    # Conversion factor to use between this and the standard unit.
    CONVERSION_FACTOR = 50

    def _from_standard(self, standard_value: Unit) -> None:
        """
        See superclass for documentation.
        """
        # Use a fake conversion factor.
        self._set_raw(standard_value.raw / self.CONVERSION_FACTOR)

    def to_standard(self) -> Unit:
        """
        See superclass for documentation.
        :return:
        """
        return _MyStandardUnit(self.raw * self.CONVERSION_FACTOR)


class TestUnit:
    """
    Tests for the Unit class.
    """

    @classmethod
    @pytest.fixture(params=[10, 5.0, np.array([1, 2, 3]), [1, 2, 3]])
    def my_unit(cls, request) -> _MyUnit:
        """
        Creates a new _MyUnit object.
        :param request: The pytest object used for parametrization. Not
        type-annotated because the type is not easily accessible.
        :return: The Unit that it created.
        """
        # Fake the unit type so it's compatible.
        _MyUnit.UNIT_TYPE = "MyUnitType"

        return _MyUnit(request.param)

    @classmethod
    @pytest.fixture(params=[10, 5.0, np.array([1, 2, 3]), [1, 2, 3]])
    def my_standard_unit(cls, request) -> _MyStandardUnit:
        """
        Creates a new _MyStandardUnit object.
        :param request: The pytest object used for parametrization. Not
        type-annotated because the type is not easily accessible.
        :return: The Unit that it created.
        """
        # Fake the unit type so it's compatible.
        _MyStandardUnit.UNIT_TYPE = "MyUnitType"

        return _MyStandardUnit(request.param)

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
        unit = _MyStandardUnit(unit_value)

        # Assert.
        # The raw value should be correct.
        np.testing.assert_array_equal(expected_value, unit.raw)

    def test_to_standard(self, my_unit: _MyUnit) -> None:
        """
        Tests that to_standard works.
        :param my_unit: The unit instance to test with.
        """
        # Arrange done in fixtures.
        # Act.
        standard_converted = my_unit.to_standard()

        # It should have been correctly converted.
        assert isinstance(standard_converted, _MyStandardUnit)
        np.testing.assert_array_equal(my_unit.raw * _MyUnit.CONVERSION_FACTOR,
                                      standard_converted.raw)

    def test_to_standard_already_converted(self, my_standard_unit:
                                           _MyStandardUnit) -> None:
        """
        Tests that to_standard works.
        :param my_standard_unit: The standard unit instance to test with.
        """
        # Arrange done in fixtures.
        # Act.
        standard_not_converted = my_standard_unit.to_standard()

        # No conversion should have happened.
        assert isinstance(standard_not_converted, _MyStandardUnit)
        np.testing.assert_array_equal(my_standard_unit.raw,
                                      standard_not_converted.raw)

    def test_init_from_other_unit(self, my_unit: _MyUnit) -> None:
        """
        Tests that we can initialize one unit from another.
        :param my_unit: The unit instance to test with.
        """
        # Arrange done in fixtures.
        # Act.
        # Create a new unit from this one.
        new_unit = _MyStandardUnit(my_unit)

        # Assert.
        # It should have done the conversion.
        np.testing.assert_array_equal(my_unit.raw * _MyUnit.CONVERSION_FACTOR,
                                      new_unit.raw)

    def test_init_wrong_type(self, my_unit: _MyUnit) -> None:
        """
        Tests that it won't let us initialize a unit from another one if the
        types don't match.
        :param my_unit: The unit instance to test with.
        """
        # Arrange.
        # Make it look like the unit has an incompatible type.
        _MyUnit.UNIT_TYPE = "MyNewUnitType"

        # Act and assert.
        with pytest.raises(UnitError):
            _MyStandardUnit(my_unit)

    def test_eq(self, my_unit: _MyUnit) -> None:
        """
        Tests that two units will compare as equal when they should.
        :param my_unit: The unit instance to test with.
        """
        # Arrange.
        # Create a unit that is the exact same as this one.
        same_unit = _MyUnit(my_unit.raw)
        # Create a unit that is different.
        different_unit = _MyUnit(my_unit.raw + 1)

        # Act and assert.
        assert my_unit == same_unit
        assert my_unit != different_unit

    def test_eq_other_unit(self, my_unit: _MyUnit) -> None:
        """
        Tests that two units will compare as equal when they are different
        units, but hold equivalent values.
        :param my_unit: The unit instance to test with.
        """
        # Arrange.
        # Create a unit that is equivalent to this one.
        same_unit = _MyStandardUnit(my_unit.raw * _MyUnit.CONVERSION_FACTOR)
        # Create a unit that is not.
        different_unit = _MyStandardUnit(my_unit.raw)

        # Act and assert.
        assert my_unit == same_unit
        assert my_unit != different_unit

    def test_name(self, my_unit: _MyUnit) -> None:
        """
        Tests that getting the name of a unit works.
        :param my_unit: The unit instance to test with.
        """
        # Arrange done in fixtures.
        # Act and assert.
        assert my_unit.name == _MyUnit.__name__

    def test_str(self, my_unit: _MyUnit) -> None:
        """
        Tests that converting a unit to a string works.
        :param my_unit: The unit instance to test with.
        """
        # Arrange done in fixtures.
        # Act.
        string_unit = str(my_unit)

        # Assert.
        assert string_unit == "{} {}".format(my_unit.raw, _MyUnit.__name__)
