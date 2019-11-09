import unittest.mock as mock

import numpy as np

import pytest

from pyunits.unitless import Unitless, UnitlessType
from pyunits.unit_interface import UnitInterface
from pyunits.unit_type import UnitType


class TestUnitless:
    """
    Tests for the Unitless class.
    """

    # Value to use for testing Unitless instances.
    _UNITLESS_VALUE = np.array([1, 2, 3])
    # Value to use for fake Units.
    _MOCK_UNIT_VALUE = np.array([2, 2, 2])

    @classmethod
    @pytest.fixture
    def unitless(cls) -> Unitless:
        """
        Creates a new Unitless instance.
        :return: The Unitless instance that it created.
        """
        return Unitless(cls._UNITLESS_VALUE)

    @classmethod
    @pytest.fixture
    def mock_unit(cls) -> mock.Mock:
        """
        Creates a fake value that implements UnitInterface.
        :return: The fake unit.
        """
        mock_unit = mock.Mock(spec=UnitInterface)

        # Set a fake raw value.
        mock_raw = mock.PropertyMock(return_value=cls._MOCK_UNIT_VALUE)
        type(mock_unit).raw = mock_raw

        return mock_unit

    def test_mul_numeric(self, unitless: Unitless) -> None:
        """
        Tests that we can multiply a Unitless value by a raw numeric value.
        :param unitless: The Unitless value to try multiplying.
        """
        # Arrange done in fixtures.
        # Act.
        product = unitless * 2.0

        # Assert.
        # It should have updated the raw value.
        assert product.raw == pytest.approx(self._UNITLESS_VALUE * 2.0)

    def test_mul_unitless(self, unitless: Unitless) -> None:
        """
        Tests that we can multiply a Unitless value by another one.
        :param unitless: The Unitless value to try multiplying.
        """
        # Arrange.
        # Create another Unitless value to multiply.
        mul_by = Unitless(2.0)

        # Act.
        product = unitless * mul_by

        # Assert.
        # It should have updated the raw value.
        assert product.raw == pytest.approx(self._UNITLESS_VALUE * 2.0)

    def test_mul_unit(self, unitless: Unitless,
                      mock_unit: mock.Mock) -> None:
        """
        Tests that we can multiply a Unitless value by another Unit.
        :param unitless: The Unitless value to try multiplying.
        :param mock_unit: The fake Unit to multiply by.
        """
        # Arrange done in fixtures.
        # Act and assert.
        with pytest.raises(NotImplementedError):
            # The fancy lambda-ing is so we don't get warnings about unused
            # variables and such.
            (lambda: unitless * mock_unit)()

    def test_div_numeric(self, unitless: Unitless) -> None:
        """
        Tests that we can divide a Unitless value by a raw numeric value.
        :param unitless: The Unitless value to try dividing.
        """
        # Arrange done in fixtures.
        # Act and assert.
        with pytest.raises(NotImplementedError, match="Division"):
            unitless / 2.0

    def test_rdiv(self, unitless: Unitless) -> None:
        """
        Tests that reversed division works.
        :param unitless: The Unitless value to try dividing by.
        """
        # Arrange done in fixtures.
        # Act.
        quotient = 2.0 / unitless

        # Assert.
        # It should have updated the raw value.
        assert quotient.raw == pytest.approx(2.0 / self._UNITLESS_VALUE)

    def test_div_unit(self, unitless: Unitless, mock_unit: mock.Mock) -> None:
        """
        Tests that we can divide a Unitless value by another unit.
        :param unitless: The Unitless value to try dividing.
        :param mock_unit: The fake Unit to divide by.
        """
        # Arrange done in fixtures.
        # Act and assert.
        with pytest.raises(NotImplementedError):
            # The fancy lambda-ing is so we don't get warnings about unused
            # variables and such.
            (lambda: unitless / mock_unit)()

    def test_div_unitless(self, unitless: Unitless) -> None:
        """
        Tests that we can divide a Unitless value by another Unitless value.
        :param unitless: The Unitless value to try dividing.
        """
        # Arrange.
        # Create another Unitless value to divide by.
        divide_by = Unitless(2.0)

        # Act.
        quotient = unitless / divide_by

        # Assert.
        # It should have gotten the correct value.
        assert quotient.raw == pytest.approx(self._UNITLESS_VALUE / 2.0)

    def test_type(self, unitless: Unitless) -> None:
        """
        Tests that a Unitless value reports the correct type.
        :param unitless: The Unitless value to get the type of.
        """
        # Arrange done in fixtures.
        # Act.
        got_type = unitless.type

        # Assert.
        assert type(got_type) == UnitlessType

    def test_type_class(self, unitless: Unitless) -> None:
        """
        Tests that a Unitless value reports the correct type class.
        :param unitless: The Unitless value to get the type of.
        """
        # Arrange done in fixtures.
        # Act.
        got_type_class = unitless.type_class

        # Assert.
        assert got_type_class == UnitlessType

    def test_is_standard(self, unitless: Unitless) -> None:
        """
        Tests that is_standard() works on Unitless values.
        :param unitless: The Unitless value to test with.
        """
        # Arrange done in fixtures.
        # Act.
        got_standard = unitless.is_standard()

        # Assert.
        # A Unitless value should always be considered standard.
        assert got_standard

    def test_to_standard(self, unitless: Unitless) -> None:
        """
        Tests that to_standard() works on Unitless values.
        :param unitless: The Unitless value to test with.
        """
        # Arrange done in fixtures.
        # Act.
        standard = unitless.to_standard()

        # Assert.
        # A Unitless value should always be considered standard.
        assert standard == unitless

    def test_raw(self, unitless: Unitless) -> None:
        """
        Tests that the raw property works on Unitless values.
        :param unitless: The Unitless value to test with.
        """
        # Arrange done in fixtures.
        # Act.
        got_raw = unitless.raw

        # Assert.
        np.testing.assert_array_almost_equal(self._UNITLESS_VALUE, got_raw)

    def test_name(self, unitless: Unitless) -> None:
        """
        Tests that the name property works on Unitless values.
        :param unitless: The Unitless value to test with.
        """
        # Arrange done in fixtures.
        # Act.
        got_name = unitless.name

        # Assert.
        assert got_name == ""

    def test_cast_to(self, unitless: Unitless) -> None:
        """
        Tests that cast_to() works on Unitless values.
        :param unitless: The Unitless value to test with.
        """
        # Arrange done in fixtures.
        # Act and assert.
        with pytest.raises(NotImplementedError, match="not ever need"):
            unitless.cast_to(mock.Mock(spec=UnitType))

    def test_add_numeric(self, unitless: Unitless) -> None:
        """
        Tests that we can add a raw numeric value to a Unitless value.
        :param unitless: The Unitless value to try adding.
        """
        # Arrange done in fixtures.
        # Act.
        unit_sum = unitless + 2.0

        # Assert.
        # It should have updated the raw value.
        assert unit_sum.raw == pytest.approx(self._UNITLESS_VALUE + 2.0)

    def test_add_unitless(self, unitless: Unitless) -> None:
        """
        Tests that we can add a Unitless value to another one.
        :param unitless: The Unitless value to try adding.
        """
        # Arrange.
        # Create another Unitless value to add.
        add_to = Unitless(2.0)

        # Act.
        unit_sum = unitless + add_to

        # Assert.
        # It should have updated the raw value.
        assert unit_sum.raw == pytest.approx(self._UNITLESS_VALUE + 2.0)

    def test_add_unit(self, unitless: Unitless,
                      mock_unit: mock.Mock) -> None:
        """
        Tests that we can add a Unitless value to another Unit.
        :param unitless: The Unitless value to try adding.
        :param mock_unit: The fake Unit to add to.
        """
        # Arrange done in fixtures.
        # Act and assert.
        with pytest.raises(NotImplementedError):
            # The fancy lambda-ing is so we don't get warnings about unused
            # variables and such.
            (lambda: unitless + mock_unit)()
