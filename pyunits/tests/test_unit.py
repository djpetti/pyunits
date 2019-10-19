from typing import NamedTuple
import unittest.mock as mock

import numpy as np

import pytest

from pyunits.exceptions import UnitError
from pyunits.types import CompoundTypeFactories, UnitValue
from pyunits import unit
from pyunits import unit_base
from . import math_op_testing
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
        :param mock_simplify: The mock simplify function.
        :param mock_wrap_numeric: The mocked WrapNumeric decorator.
        """
        standard_unit: unit.Unit
        unit: unit.Unit
        mock_type: mock.Mock
        mock_mul: mock.Mock
        mock_div: mock.Mock
        mock_simplify: mock.Mock
        mock_wrap_numeric: mock.Mock

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

        with mock.patch(unit_base.__name__ + ".unit_analysis.simplify") as \
                mock_simplify, \
                mock.patch(unit_base.__name__ + ".WrapNumeric") as \
                mock_wrap_numeric:
            # Make the wrapper not change the function.
            mock_wrap_numeric.return_value.side_effect = lambda x: x

            yield cls.UnitConfig(unit=my_unit, standard_unit=standard_unit,
                                 mock_type=unit_type,
                                 mock_mul=mock_mul,
                                 mock_div=mock_div,
                                 mock_simplify=mock_simplify,
                                 mock_wrap_numeric=mock_wrap_numeric)

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
        same_unit = MyUnit(mock_type, config.unit.raw)
        # Create a unit that is different.
        different_unit = MyUnit(mock_type, config.unit.raw + 1)

        # Act and assert.
        assert config.unit.equals(same_unit)
        assert not config.unit.equals(different_unit)

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
        assert config.unit.equals(same_unit)
        assert not config.unit.equals(different_unit)

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
        # It should have created a new compound unit type.
        config.mock_mul.assert_called_once_with(config.mock_type,
                                                config.mock_type)
        compound_unit_type = config.mock_mul.return_value

        # It should have created the compound unit.
        compound_unit_type.apply_to.assert_called_once_with(config.unit,
                                                            converted)
        compound_unit = compound_unit_type.apply_to.return_value

        # It should have simplified the compound unit.
        config.mock_simplify.assert_called_once_with(compound_unit, mock.ANY)
        # It should have returned the compound unit.
        assert product == config.mock_simplify.return_value

    @pytest.mark.parametrize("simplify", [False, True])
    def test_mul_incompatible_unit(self, config: UnitConfig,
                                   simplify: bool) -> None:
        """
        Tests that the multiplication operation works correctly when a unit is
        multiplied by another with an incompatible type.
        :param config: The configuration to use.
        :param simplify: Whether we should simulate simplification of the
        resulting type.
        """
        # Arrange, act and assert.
        math_op_testing.test_mul_incompatible_unit(
            config.unit, config.mock_type, config.mock_mul,
            config.mock_simplify,
            simplify=simplify,
            passes_op=False
        )

    @pytest.mark.parametrize("simplify", [False, True])
    def test_div_incompatible_unit(self, config: UnitConfig,
                                   simplify: bool) -> None:
        """
        Tests that the division operation works correctly when a unit is
        divided by another with an incompatible type.
        :param config: The configuration to use.
        :param simplify: Whether we should simulate simplification of the
        resulting type.
        """
        # Arrange, act and assert.
        math_op_testing.test_div_incompatible_unit(
            config.unit, config.mock_type, config.mock_div,
            config.mock_simplify,
            simplify=simplify,
            passes_op=False
        )
