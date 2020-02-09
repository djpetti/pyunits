from collections import OrderedDict as Od
from typing import Dict, NamedTuple
import functools
import unittest.mock as mock

import pytest

from pyunits.compound_units import pretty_print
from pyunits.types import RequestType
from ...tests.testing_types import UnitFactory


class PrettyNameTest(NamedTuple):
    """
    Represents a single test-case for the pretty_name() function.
    :param mock_unit: The mocked Unit that we will try printing.
    :param mock_numerator: The mocked numerator of the flattened unit.
    :param mock_denominator: The mocked denominator of the flattened unit.
    :param expected_name: The expected name that should be the result.
    """
    mock_unit: mock.Mock
    mock_numerator: Dict[mock.Mock, int]
    mock_denominator: Dict[mock.Mock, int]
    expected_name: str


class ConfigForTests(NamedTuple):
    """
    Encapsulates standard configuration for most tests.
    :param mock_flatten: The mocked unit_analysis.flatten() function.
    """
    mock_flatten: mock.Mock


# Total number of tests for pretty_name() that we have.
_NUM_PRETTY_NAME_TESTS = 5


@pytest.fixture(params=range(_NUM_PRETTY_NAME_TESTS),
                ids=["single_unit", "squared_unit", "simple_denominator",
                     "complex", "no_numerator"])
def pretty_name_test(request: RequestType, unit_factory: UnitFactory
                     ) -> PrettyNameTest:
    """
    Generates PrettyNameTests.
    :param request: The PyTest request object to use for parametrization.
    :param unit_factory: The factory to use for creating fake units.
    :return: The PrettyNameTest that it generated.
    """
    # A fake unit to use for all tests. It doesn't really matter what it is
    # because we mock the result of flatten().
    mock_unit = unit_factory("TestUnit")

    unit1 = unit_factory("a", raw=1.0)
    unit2 = unit_factory("b", raw=2.0)
    unit3 = unit_factory("c", raw=3.0)

    test_class = functools.partial(PrettyNameTest, mock_unit=mock_unit)
    # The list of tests to run. We use ordered dicts for the numerator and
    # denominator, because the results depend on the order in which these
    # dicts are iterated through, and we want them to be consistent.
    tests = [
        # A simple case where the unit is not compound.
        test_class(mock_numerator={unit1: 1}, mock_denominator={},
                   expected_name="a"),
        # A simple case where the unit is squared.
        test_class(mock_numerator={unit1: 2}, mock_denominator={},
                   expected_name="a^2"),
        # A simple case with a denominator.
        test_class(mock_numerator={unit1: 1}, mock_denominator={unit2: 1},
                   expected_name=" a \n"
                                 "---\n"
                                 " b "),
        # A more complicated case with everything.
        test_class(mock_numerator=Od({unit1: 3, unit2: 1}),
                   mock_denominator={unit3: 2},
                   expected_name=" (a^3)(b) \n"
                                 "----------\n"
                                 "   c^2    "),
        # A case with no numerator.
        test_class(mock_numerator={}, mock_denominator=Od({unit1: 2, unit2: 1}),
                   expected_name="    1     \n"
                                 "----------\n"
                                 " (a^2)(b) "),
    ]

    assert len(tests) == _NUM_PRETTY_NAME_TESTS, "_NUM_PRETTY_NAME_TESTS " \
                                                 " should be updated to " \
                                                 "reflect the number of " \
                                                 "parametrized tests."
    return tests[request.param]


@pytest.fixture
def config() -> ConfigForTests:
    """
    Generates configuration for tests.
    :return: The configuration that it generated.
    """
    with mock.patch(pretty_print.__name__ + ".flatten"
                    ) as mock_flatten:
        yield ConfigForTests(mock_flatten=mock_flatten)
        # Finalization done implicitly upon exit from context manager.


def test_pretty_name(config: ConfigForTests, pretty_name_test: PrettyNameTest
                     ) -> None:
    """
    Tests that pretty_name() works.
    :param config: The configuration to use for testing.
    :param pretty_name_test: The specific test case.
    """
    # Arrange.
    # Set the mocked version of flatten() to return the correct thing.
    config.mock_flatten.return_value = (pretty_name_test.mock_numerator,
                                        pretty_name_test.mock_denominator)

    # Act.
    name = pretty_print.pretty_name(pretty_name_test.mock_unit)

    # Assert.
    assert name == pretty_name_test.expected_name

    # It should have flattened the input.
    config.mock_flatten.assert_called_once_with(pretty_name_test.mock_unit)
