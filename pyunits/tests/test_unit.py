from typing import NamedTuple
import unittest.mock as mock

import numpy as np

import pytest

from pyunits.exceptions import UnitError
from pyunits.types import CompoundTypeFactories, UnitValue
from pyunits.tests.testing_types import UnitFactory
from pyunits import unit
from .helpers import MyUnit, MyStandardUnit


class TestUnit:
    """
    Tests for the Unit class.
    """

    class UnitConfig(NamedTuple):
        """
        Configuration for the tests.
        :param standard_unit: The standard unit instance to test with.
        :param unit: The unit instance under test.
        :param mock_type: The UnitType that the Unit was constructed with.
        :param mock_mul: The mock compound_units.Mul function.
        :param mock_div: The mock compound_units.Div function.
        :param mock_do_mul: The mocked do_mul() function.
        :param mock_do_div: The mocked do_div() function.
        :param mock_do_add: The mocked do_add() function.
        """
        standard_unit: unit.Unit
        other_unit: unit.Unit
        mock_type: mock.Mock
        mock_mul: mock.Mock
        mock_div: mock.Mock
        mock_do_mul: mock.Mock
        mock_do_div: mock.Mock
        mock_do_add: mock.Mock

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

        # Replace the compound type factories with mocks.
        mock_mul = mock.Mock()
        mock_div = mock.Mock()
        unit.Unit.COMPOUND_TYPE_FACTORIES = CompoundTypeFactories(mul=mock_mul,
                                                                  div=mock_div)

        with mock.patch(unit.__name__ + ".do_mul") as mock_do_mul, \
                mock.patch(unit.__name__ + ".do_div") as mock_do_div, \
                mock.patch(unit.__name__ + ".do_add") as mock_do_add:
            yield cls.UnitConfig(other_unit=my_unit,
                                 standard_unit=standard_unit,
                                 mock_type=unit_type,
                                 mock_mul=mock_mul,
                                 mock_div=mock_div,
                                 mock_do_mul=mock_do_mul,
                                 mock_do_div=mock_do_div,
                                 mock_do_add=mock_do_add)

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
        this_unit = MyStandardUnit(unit_type, unit_value)

        # Assert.
        # The raw value should be correct.
        np.testing.assert_array_equal(expected_value, this_unit.raw)

    def test_to_standard(self, config: UnitConfig) -> None:
        """
        Tests that to_standard works.
        :param config: The configuration to use for testing.
        """
        # Arrange done in fixtures.
        # Act.
        standard_converted = config.other_unit.to_standard()

        # It should have been correctly converted.
        assert isinstance(standard_converted, MyStandardUnit)
        np.testing.assert_array_equal(config.other_unit.raw *
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
        new_unit = MyStandardUnit(config.mock_type, config.other_unit)

        # Assert.
        # It should have done the conversion.
        np.testing.assert_array_equal(config.other_unit.raw *
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
            MyStandardUnit(my_type, config.other_unit)

    def test_equals(self, config: UnitConfig) -> None:
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
        same_unit = MyUnit(mock_type, config.other_unit.raw)
        # Create a unit that is different.
        different_unit = MyUnit(mock_type, config.other_unit.raw + 1)

        # Act and assert.
        assert config.other_unit.equals(same_unit)
        assert not config.other_unit.equals(different_unit)

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
                                   config.other_unit.raw * MyUnit.CONVERSION_FACTOR)
        # Create a unit that is not.
        different_unit = MyStandardUnit(mock_type, config.other_unit.raw)

        # Act and assert.
        assert config.other_unit.equals(same_unit)
        assert not config.other_unit.equals(different_unit)

    @pytest.mark.parametrize("is_standard", [False, True],
                             ids=["not_standard", "standard"])
    def test_is_standard(self, is_standard: bool) -> None:
        """
        Tests that is_standard() works.
        :param is_standard: Whether to test this on the standard unit.
        """
        # Arrange.
        unit_class = MyUnit
        if is_standard:
            unit_class = MyStandardUnit

        # Act.
        standard = unit_class.is_standard()

        # Assert.
        assert standard == is_standard

    def test_name(self, config: UnitConfig) -> None:
        """
        Tests that getting the name of a unit works.
        :param config: The configuration to use for testing.
        """
        # Arrange done in fixtures.
        # Act and assert.
        assert config.other_unit.name == MyUnit.__name__

    def test_str(self, config: UnitConfig) -> None:
        """
        Tests that converting a unit to a string works.
        :param config: The configuration to use for testing.
        """
        # Arrange done in fixtures.
        # Act.
        string_unit = str(config.other_unit)

        # Assert.
        assert string_unit == "{} {}".format(config.other_unit.raw,
                                             MyUnit.__name__)

    def test_cast_to(self, config: UnitConfig) -> None:
        """
        Tests that the casting helper works.
        :param config: The configuration to use for testing.
        """
        # Arrange.
        # Mock an output unit class.
        out_unit = mock.MagicMock()

        # Act.
        got_unit = config.other_unit.cast_to(out_unit)

        # Assert.
        out_type = out_unit.__class__
        # It should have performed the cast.
        config.mock_type.as_type.assert_called_once_with(config.other_unit,
                                                         out_type)

        # It should have initialized the new unit instance.
        out_unit.assert_called_once_with(config.mock_type.as_type.return_value)
        assert got_unit == out_unit.return_value

    @pytest.mark.parametrize("is_reversed", [False, True],
                             ids=["normal", "reversed"])
    def test_mul(self, config: UnitConfig, unit_factory: UnitFactory,
                 is_reversed: bool) -> None:
        """
        Tests that we can multiply the unit by another.
        :param config: The configuration to use for testing.
        :param unit_factory: The UnitFactory to use for creating test units.
        :param is_reversed: Whether to use reversed multiplication for the test.
        """
        # Arrange.
        # Create a fake unit to multiply by.
        mul_by = unit_factory("Unit1", 1.0)

        # Act.
        if not is_reversed:
            product = config.other_unit * mul_by
        else:
            product = mul_by * config.other_unit

        # Assert.
        # It should do the same thing either way because for multiplication,
        # order doesn't matter.
        config.mock_do_mul.assert_called_once_with(mock.ANY,
                                                   config.other_unit,
                                                   mul_by)
        assert product == config.mock_do_mul.return_value

    @pytest.mark.parametrize("is_reversed", [False, True],
                             ids=["normal", "reversed"])
    def test_div(self, config: UnitConfig, unit_factory: UnitFactory,
                 is_reversed: bool) -> None:
        """
        Tests that we can divide the compound unit by another.
        :param config: The configuration to use for testing.
        :param unit_factory: The UnitFactory to use for creating test units.
        :param is_reversed: Whether to use reversed multiplication for the test.
        """
        # Arrange.
        # Create a fake unit to divide by.
        div_by = unit_factory("Unit1", 1.0)

        # Act.
        if not is_reversed:
            quotient = config.other_unit / div_by
            do_div_args = (config.other_unit, div_by)
        else:
            quotient = div_by / config.other_unit
            do_div_args = (div_by, config.other_unit)

        # Assert.
        config.mock_do_div.assert_called_once_with(mock.ANY,
                                                   *do_div_args)
        assert quotient == config.mock_do_div.return_value

    def test_negation(self, config: UnitConfig) -> None:
        """
        Tests that a Unit can be negated.
        :param config: The configuration to use for testing.
        """
        # Arrange.
        # Mock the UnitType invocation so it returns a new unit.
        config.mock_type.side_effect = lambda x: MyUnit(config.mock_type, x)

        # Act.
        negated = -config.other_unit

        # Assert.
        np.testing.assert_array_equal(-config.other_unit.raw, negated.raw)

    @pytest.mark.parametrize("is_reversed", [False, True],
                             ids=["normal", "reversed"])
    def test_add(self, config: UnitConfig, unit_factory: UnitFactory,
                 is_reversed: bool) -> None:
        """
        Tests that we can add the unit to another.
        :param config: The configuration to use for testing.
        :param unit_factory: The UnitFactory to use for creating test units.
        :param is_reversed: Whether to use reversed multiplication for the test.
        """
        # Arrange.
        # Create a fake unit to add.
        add_to = unit_factory("Unit1", 1.0)

        # Act.
        if not is_reversed:
            unit_sum = config.other_unit + add_to
        else:
            unit_sum = add_to + config.other_unit

        # Assert.
        # It should do the same thing either way because for addition,
        # order doesn't matter.
        config.mock_do_add.assert_called_once_with(config.other_unit,
                                                   add_to)
        assert unit_sum == config.mock_do_add.return_value

    def test_sub(self, config: UnitConfig) -> None:
        """
        Tests that the subtraction operation works on this unit.
        :param config: The configuration to use for testing.
        """
        # Arrange.
        # Create a fake unit to subtract. We can't use the UnitFactory here,
        # because it needs to support negation.
        to_subtract = mock.MagicMock(spec=unit.Unit)

        # Act.
        diff = config.other_unit - to_subtract

        # Assert.
        # It should have negated the value.
        to_subtract.__neg__.assert_called_once_with()
        negated = to_subtract.__neg__.return_value

        # It should have used do_add() on a negated value.
        config.mock_do_add.asssert_called_once_with(config.other_unit, negated)
        assert diff == config.mock_do_add.return_value

    def test_sub_reversed(self, config: UnitConfig,
                          unit_factory: UnitFactory) -> None:
        """
        Tests that the reversed subtraction operation works on this unit.
        :param config: The configuration to use for testing.
        :param unit_factory: The UnitFactory to use for creating test units.
        """
        # Arrange.
        # Create a fake unit to subtract.
        to_subtract = unit_factory("Unit1", 1.0)

        # Mock the negation.
        config.mock_type.side_effect = lambda x: MyUnit(config.mock_type, x)

        # Act.
        diff = to_subtract - config.other_unit

        # Assert.
        config.mock_do_add.asssert_called_once_with(-config.other_unit,
                                                    to_subtract)
        assert diff == config.mock_do_add.return_value
