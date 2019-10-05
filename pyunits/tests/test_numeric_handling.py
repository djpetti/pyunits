from typing import Any, Dict, Iterable, NamedTuple, Optional
import unittest.mock as mock

import numpy as np

import pytest

from pyunits import numeric_handling as nh
from pyunits.unitless import Unitless


class TestWrapNumeric:
    """
    Tests for the WrapNumeric class.
    """

    class WrapperTest(NamedTuple):
        """
        Represents a particular test case for the wrapper.
        :param args: The arguments to pass to the wrapped function.
        :param kwargs: The keyword arguments to pass to the wrapped function.
        """
        args: Iterable = ()
        kwargs: Dict[str, Any] = {}

    @pytest.fixture
    def mock_unitless(self) -> mock.Mock:
        """
        Mocks the Unitless class.
        :return: The mocked Unitless class.
        """
        with mock.patch(nh.__name__ + ".Unitless") as mock_unitless:
            yield mock_unitless

            # Finalization done implicitly upon exit from context manager.

    @pytest.mark.parametrize("test_case", [
        WrapperTest(args=(1, "foo")),
        WrapperTest(args=(3.14, "foo")),
        WrapperTest(args=(np.array([1, 2, 3]), "foo")),
        WrapperTest(args=((1, 2, 3), "foo")),
        WrapperTest(args=([1, 2, 3], "foo")),
    ])
    def test_simple_function(self, test_case: WrapperTest,
                             mock_unitless: mock.Mock) -> None:
        """
        Tests that the wrapper works for a simple function with no keyword
        arguments.
        :param test_case: The test case to use.
        :param mock_unitless: The mocked Unitless class.
        """
        # Arrange.
        # This is the function we'll use for testing.
        @nh.WrapNumeric("numeric_arg")
        def test_func(numeric_arg: Unitless, str_arg: str) -> None:
            # Assert.
            # The first argument should always be wrapped in a Unitless
            # instance.
            assert numeric_arg == mock_unitless.return_value
            # The second argument should not be touched.
            assert type(str_arg) is str

        # Act.
        test_func(*test_case.args, **test_case.kwargs)

    @pytest.mark.parametrize("test_case", [
        WrapperTest(kwargs={"numeric_arg": 1}),
        WrapperTest(args=(1,)),
        WrapperTest(args=(1, "foo")),
        WrapperTest(kwargs={"numeric_arg": 3.14, "other_arg": "hello"}),
        WrapperTest(kwargs={"numeric_arg": np.array([1, 2, 3])}),
        WrapperTest(kwargs={"numeric_arg": (1, 2, 3)}),
        WrapperTest(kwargs={"numeric_arg": [1, 2, 3]}),
        WrapperTest(),
    ])
    def test_keyword_args(self, test_case: WrapperTest,
                          mock_unitless: mock.Mock) -> None:
        """
        Tests that the wrapper works for functions with keyword arguments.
        :param test_case: The test case to use.
        :param mock_unitless: The mocked Unitless class.
        """
        # Arrange.
        # This is the function we'll use for testing.
        @nh.WrapNumeric("numeric_arg")
        def test_func(numeric_arg: Optional[Unitless] = None,
                      other_arg: str = "") -> None:
            # Assert.
            # The keyword argument should either be wrapped in a Unitless
            # instance, or None.
            assert numeric_arg == mock_unitless.return_value or \
                   numeric_arg is None
            # The other argument should never have been touched.
            assert type(other_arg) is str

        # Act.
        test_func(*test_case.args, **test_case.kwargs)

    @pytest.mark.parametrize("test_case", [
        WrapperTest(args=(1,), kwargs={"keyword_arg": 2}),
        WrapperTest(args=(1,), kwargs={"keyword_arg": 3.14}),
        WrapperTest(args=(1,), kwargs={"keyword_arg": np.array([1, 2, 3])}),
        WrapperTest(args=(1,), kwargs={"keyword_arg": (1, 2, 3)}),
        WrapperTest(args=(1,), kwargs={"keyword_arg": [1, 2, 3]}),
    ])
    def test_required_keyword_args(self, test_case: WrapperTest,
                                   mock_unitless: mock.Mock) -> None:
        """
        Tests that the wrapper works for functions with required keyword
        arguments.
        :param test_case: The test case to use.
        :param mock_unitless: The mocked Unitless class.
        """
        # Arrange.
        # This is the function we'll use for testing.
        @nh.WrapNumeric("pos_arg", "keyword_arg")
        def test_func(pos_arg: Unitless, *, keyword_arg: Unitless) -> None:
            # Assert.
            # Both arguments should be wrapped in unitless instances.
            assert pos_arg == mock_unitless.return_value
            assert keyword_arg == mock_unitless.return_value

        # Act.
        test_func(*test_case.args, **test_case.kwargs)

    @pytest.mark.parametrize("test_case", [
        WrapperTest(args=({},), kwargs={"keyword_arg": "hello"}),
        WrapperTest(args=({"foo": 1},), kwargs={"keyword_arg": ""}),
    ])
    def test_not_converted(self, test_case: WrapperTest,
                           mock_unitless: mock.Mock) -> None:
        """
        Tests that it doesn't convert values that are the wrong type.
        :param test_case: The test case to use.
        :param mock_unitless: The mocked Unitless class.
        """
        # Arrange.
        # This is the function we'll use for testing.
        @nh.WrapNumeric("pos_arg", "keyword_arg")
        def test_func(pos_arg: Dict, keyword_arg: str) -> None:
            # Assert.
            # Both arguments should not have been touched.
            assert type(pos_arg) == dict
            assert type(keyword_arg) == str

        # Act.
        test_func(*test_case.args, **test_case.kwargs)
