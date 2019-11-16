import unittest.mock as mock

import numpy as np

import pytest

from pyunits.exceptions import UnitError
from pyunits.types import RequestType
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
    @pytest.fixture(params=[_UNITLESS_VALUE, Unitless(_UNITLESS_VALUE)],
                    ids=["init_raw_numeric", "init_unitless"])
    def unitless(cls, request: RequestType) -> Unitless:
        """
        Creates a new Unitless instance.
        :param request: The PyTest Request object to use for parametrization.
        :return: The Unitless instance that it created.
        """
        return Unitless(request.param)

    @classmethod
    @pytest.fixture
    def mock_unit(cls) -> mock.MagicMock:
        """
        Creates a fake value that implements UnitInterface.
        :return: The fake unit.
        """
        mock_unit = mock.MagicMock(spec=UnitInterface)

        # Set a fake raw value.
        mock_raw = mock.PropertyMock(return_value=cls._MOCK_UNIT_VALUE)
        type(mock_unit).raw = mock_raw

        return mock_unit

    def test_init_other_unit(self, mock_unit: mock.Mock) -> None:
        """
        Tests that Unitless refuses to convert implicitly from another Unit.
        :param mock_unit: The Unit to try to convert from.
        """
        # Arrange.
        # Make it look like it is not another Unitless value.
        mock_unit.type.is_compatible.return_value = False

        # Act and assert.
        with pytest.raises(UnitError, match="explicitly take"):
            Unitless(mock_unit)

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
        # Act.
        result = unitless * mock_unit

        # Assert.
        # The multiplication operator for the Unitless instance should have
        # returned NotImplemented, which should have caused it to use the
        # reflected multiplication operator on the mock.
        mock_unit.__rmul__.assert_called_once_with(unitless)
        assert result == mock_unit.__rmul__.return_value

    def test_div_numeric(self, unitless: Unitless) -> None:
        """
        Tests that we can divide a Unitless value by a raw numeric value.
        :param unitless: The Unitless value to try dividing.
        """
        # Arrange done in fixtures.
        # Act.
        result = unitless / 2.0

        # Assert.
        np.testing.assert_array_almost_equal(result.raw,
                                             self._UNITLESS_VALUE / 2)

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
        # Act.
        result = unitless / mock_unit

        # Assert.
        # The division operator for the Unitless instance should have
        # returned NotImplemented, which should have caused it to use the
        # reflected division operator on the mock.
        mock_unit.__rtruediv__.assert_called_once_with(unitless)
        assert result == mock_unit.__rtruediv__.return_value

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
        # Act.
        result = unitless + mock_unit

        # Assert.
        # The addition operator for the Unitless instance should have
        # returned NotImplemented, which should have caused it to use the
        # reflected addition operator on the mock.
        mock_unit.__radd__.assert_called_once_with(unitless)
        assert result == mock_unit.__radd__.return_value
