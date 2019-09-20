from collections import OrderedDict as od
from typing import Callable, Dict, FrozenSet, NamedTuple, Tuple, Union
import functools
import unittest.mock as mock

import pytest

from pyunits.compound_units.compound_unit_type import CompoundUnitType
from pyunits.compound_units.operations import Operation
from pyunits.compound_units import unit_analysis
from pyunits.types import RequestType
from pyunits.unit_type import UnitType


class TestUnitAnalysis:
    """
    Tests for the unit_analysis module.
    """

    # Type alias for a function that makes UnitTypes.
    UnitTypeFactory = Callable[[str], UnitType]
    # Type alias for a function that makes CompoundUnitTypes.
    CompoundTypeFactory = Callable[[Operation, UnitType, UnitType],
                                   CompoundUnitType]
    # Type that we use for fake compound units. It allows us to make
    # argument order matter for division, but not for multiplication.
    FakeCompoundType = Tuple[Operation, Union[Tuple[UnitType, UnitType],
                                              FrozenSet[UnitType]]]
    # Type alias for a function that makes FakeCompoundTypes.
    FakeCompoundTypeFactory = Callable[[Operation, UnitType, UnitType],
                                       FakeCompoundType]

    class FlattenTest(NamedTuple):
        """
        Represents a single test-case for the flatten() function.
        :param input_type: The input UnitType to try flattening.
        :param expected_numerator: The expected numerator set.
        :param expected_denominator: The expected denominator set.
        """
        input_type: UnitType
        expected_numerator: Dict[UnitType, int]
        expected_denominator: Dict[UnitType, int]

    class UnFlattenTest(NamedTuple):
        """
        Represents a single test-case for the un_flatten() function.
        :param numerator: The numerator set to un-flatten.
        :param denominator: The denominator set to un-flatten.
        :param expected_compound: The expected CompoundUnitType that would
        result.
        """
        numerator: Dict[UnitType, int]
        denominator: Dict[UnitType, int]
        expected_compound: UnitType

    class SimplifyTest(NamedTuple):
        """
        Represents a single test-case for the simplify() function.
        :param to_simplify: The CompoundUnitType to simplify.
        :param simplified: The expected simplified type.
        """
        to_simplify: CompoundUnitType
        simplified: CompoundUnitType

    @classmethod
    @pytest.fixture
    def unit_type_factory(cls) -> UnitTypeFactory:
        """
        A factory that creates a new (mock) UnitType object. It takes a string
        name for the class. If the function is called twice with the same
        string, it will return two instances of the same fake UnitType subclass.
        :return: Function that returns a new UnitType when called.
        """
        def _unit_type_factory_impl(class_name: str) -> UnitType:
            # Create the fake UnitType subclass.
            subclass = type(class_name, (UnitType,), {})
            return mock.Mock(spec=subclass)

        return _unit_type_factory_impl

    @classmethod
    @pytest.fixture
    def compound_type_factory(cls) -> CompoundTypeFactory:
        """
        A factory that creates a new (mock) CompoundUnitType object, when passed
        the same arguments as the actual constructor of CompoundUnitType.
        :return: Function that returns a new CompoundUnitType when called.
        """
        def _compound_type_factory_impl(operation: Operation,
                                        left_type: UnitType,
                                        right_type: UnitType
                                        ) -> CompoundUnitType:
            mock_type = mock.Mock(spec=CompoundUnitType)

            # Set the operation, left and right properties.
            op_property = mock.PropertyMock(return_value=operation)
            type(mock_type).operation = op_property

            left_property = mock.PropertyMock(return_value=left_type)
            type(mock_type).left = left_property

            right_property = mock.PropertyMock(return_value=right_type)
            type(mock_type).right = right_property

            return mock_type

        return _compound_type_factory_impl

    @classmethod
    @pytest.fixture
    def fake_compound_type_factory(cls) -> FakeCompoundTypeFactory:
        """
        A factory that creates a new object that can be substituted for
        CompoundUnitTypes, and allows us to make a direct equality comparison.
        It also mocks out the CompoundUnitType constructor so that calling it
        calls this factory function instead.
        :return: Function that creates fake CompoundUnitTypes.
        """
        def fake_compound_type_factory_impl(operation: Operation,
                                            left: UnitType,
                                            right: UnitType
                                            ) -> cls.FakeCompoundType:
            arg_group = (left, right)
            if operation == Operation.MUL:
                # Argument order doesn't matter.
                arg_group = frozenset(arg_group)
            return operation, arg_group

        # Mock the CompoundUnitType constructor so that it uses our fake
        # Tuple mocks.
        with mock.patch(unit_analysis.__name__ + ".CompoundUnitType") as \
                mock_compound_type:
            mock_compound_type.side_effect = fake_compound_type_factory_impl

            yield fake_compound_type_factory_impl

            # Finalization done implicitly upon exit from context manager.

    @classmethod
    @pytest.fixture(params=range(7))
    def flatten_test_case(cls, request: RequestType,
                          unit_type_factory: UnitTypeFactory,
                          compound_type_factory: CompoundTypeFactory
                          ) -> FlattenTest:
        """
        Creates a new FlattenTest object to try.
        :param request: The request to use for parametrization.
        :param unit_type_factory: The factory to use for creating UnitTypes.
        :param compound_type_factory: The factory to use for creating
        CompoundUnitTypes.
        :return: The FlattenTest to use.
        """
        test_index = request.param

        type1 = unit_type_factory("UnitType1")
        type2 = unit_type_factory("UnitType2")
        type3 = unit_type_factory("UnitType3")

        mul_factory = functools.partial(compound_type_factory, Operation.MUL)
        div_factory = functools.partial(compound_type_factory, Operation.DIV)

        # The list of tests that we want to perform.
        flatten_tests = [
            # When we try to flatten something already flat.
            cls.FlattenTest(input_type=type1,
                            expected_numerator={type1: 1},
                            expected_denominator={}),
            # When we try to flatten something simple.
            cls.FlattenTest(input_type=div_factory(type1, type2),
                            expected_numerator={type1: 1},
                            expected_denominator={type2: 1}),
            cls.FlattenTest(input_type=mul_factory(type1, type2),
                            expected_numerator={type1: 1, type2: 1},
                            expected_denominator={}),
            cls.FlattenTest(input_type=mul_factory(type1, type1),
                            expected_numerator={type1: 2},
                            expected_denominator={}),
            # When we try to flatten something nested.
            cls.FlattenTest(input_type=div_factory(mul_factory(type1, type2),
                                                   mul_factory(type2, type3)),
                            expected_numerator={type1: 1, type2: 1},
                            expected_denominator={type2: 1, type3: 1}),
            cls.FlattenTest(input_type=div_factory(mul_factory(type1, type1),
                                                   div_factory(
                                                       div_factory(type2,
                                                                   type1),
                                                       mul_factory(type3,
                                                                   type3),
                                                   )),
                            expected_numerator={type1: 3, type3: 2},
                            expected_denominator={type2: 1}),
            cls.FlattenTest(input_type=mul_factory(type1,
                                                   div_factory(type2, type3)),
                            expected_numerator={type1: 1, type2: 1},
                            expected_denominator={type3: 1}),
        ]

        return flatten_tests[test_index]

    @classmethod
    @pytest.fixture(params=range(7))
    def simplify_test_case(cls, request: RequestType,
                           unit_type_factory: UnitTypeFactory,
                           compound_type_factory: CompoundTypeFactory,
                           fake_compound_type_factory: FakeCompoundTypeFactory
                           ) -> SimplifyTest:
        """
        Creates a new SimplifyTest object to try.
        :param request: The request to use for parametrization.
        :param unit_type_factory: The factory to use for creating UnitTypes.
        :param compound_type_factory: The factory to use for creating
        CompoundUnitTypes.
        :param fake_compound_type_factory: The factory to use for creating
        fake CompoundUnitTypes.
        :return: The SimplifyTest to use.
        """
        test_index = request.param

        type1 = unit_type_factory("UnitType1")
        type2 = unit_type_factory("UnitType2")
        type3 = unit_type_factory("UnitType3")
        type4 = unit_type_factory("UnitType4")

        # We use fake CompoundUnitTypes for the output from simplify() and real
        # ones for the input.
        fake_mul = functools.partial(fake_compound_type_factory,
                                     Operation.MUL)
        fake_div = functools.partial(fake_compound_type_factory,
                                     Operation.DIV)
        real_mul = functools.partial(compound_type_factory, Operation.MUL)
        real_div = functools.partial(compound_type_factory, Operation.DIV)

        simplified1 = real_mul(type1, type2)
        simplified2 = real_div(real_mul(type1, type2), real_mul(type3, type4))
        simplified3 = real_div(type1, real_div(type2, type1))

        # The list of tests that we want to perform. We use ordered dict for
        # the attributes here because the iteration order can change the results
        # of un_flatten(), and we want them to be consistent.
        simplify_tests = [
           # When no simplification is necessary. In these cases, we should
           # just return the input.
           cls.SimplifyTest(to_simplify=simplified1, simplified=simplified1),
           cls.SimplifyTest(to_simplify=simplified2, simplified=simplified2),
           cls.SimplifyTest(to_simplify=simplified3, simplified=simplified3),
           # Simple simplification cases.
           cls.SimplifyTest(to_simplify=real_div(real_mul(type1, type2),
                                                 real_mul(type2, type3)),
                            simplified=fake_div(type1, type3)),
           cls.SimplifyTest(to_simplify=real_div(real_mul(
               real_mul(type1, type1),
               type2,
           ), type1),
                            simplified=fake_mul(type1, type2)),
           # Nested divisions.
           cls.SimplifyTest(to_simplify=real_div(real_div(type1, type2),
                                                 real_div(type3, type2)),
                            simplified=fake_div(type1, type3)),
           cls.SimplifyTest(to_simplify=real_div(real_div(
               type2, real_mul(type1, type1)),
               real_div(type3, type1)),
                            simplified=fake_div(type2, fake_mul(type1, type3))),
        ]

        return simplify_tests[test_index]

    @classmethod
    @pytest.fixture(params=range(6))
    def un_flatten_test_case(cls, request: RequestType,
                             unit_type_factory: UnitTypeFactory,
                             fake_compound_type_factory: FakeCompoundTypeFactory
                             ) -> UnFlattenTest:
        """
        Creates a new UnFlattenTest object to try.
        :param request: The request to use for parametrization.
        :param unit_type_factory: The factory to use for creating UnitTypes.
        :param fake_compound_type_factory: The factory to use for creating
        fake CompoundUnitTypes.
        :return: The UnFlattenTest to use.
        """
        test_index = request.param

        type1 = unit_type_factory("UnitType1")
        type2 = unit_type_factory("UnitType2")
        type3 = unit_type_factory("UnitType3")
        type4 = unit_type_factory("UnitType4")

        mul_factory = functools.partial(fake_compound_type_factory,
                                        Operation.MUL)
        div_factory = functools.partial(fake_compound_type_factory,
                                        Operation.DIV)

        # The list of tests that we want to perform. We use ordered dict for
        # the attributes here because the iteration order can change the results
        # of un_flatten(), and we want them to be consistent.
        un_flatten_tests = [
            # When we have no denominator.
            cls.UnFlattenTest(numerator=od({type1: 1}), denominator={},
                              expected_compound=type1),
            cls.UnFlattenTest(numerator=od({type1: 2}), denominator={},
                              expected_compound=mul_factory(type1, type1)),
            # When we do have a denominator.
            cls.UnFlattenTest(numerator=od({type1: 1}),
                              denominator=od({type2: 2}),
                              expected_compound=div_factory(
                                  type1,
                                  mul_factory(type2, type2))),
            cls.UnFlattenTest(numerator=od({type1: 1, type2: 1}),
                              denominator=od({type3: 2}),
                              expected_compound=div_factory(
                                  mul_factory(type1, type2),
                                  mul_factory(type3, type3),
                              )),
            # When we have nested products.
            cls.UnFlattenTest(numerator=od({type1: 2, type2: 2}),
                              denominator=od({type3: 2, type4: 2}),
                              expected_compound=div_factory(
                                 mul_factory(
                                     mul_factory(type1, type1),
                                     mul_factory(type2, type2)
                                 ),
                                 mul_factory(
                                     mul_factory(type3, type3),
                                     mul_factory(type4, type4)
                                 )
                              )),
            cls.UnFlattenTest(numerator=od({type1: 2, type2: 1}),
                              denominator=od({type3: 1, type4: 2}),
                              expected_compound=div_factory(
                                  mul_factory(
                                      mul_factory(type1, type2),
                                      type1,
                                  ),
                                  mul_factory(
                                      mul_factory(type4, type4),
                                      type3,
                                  )
                              )),
        ]

        return un_flatten_tests[test_index]

    def test_flatten(self, flatten_test_case: FlattenTest) -> None:
        """
        Tests that the flatten() function works.
        :param flatten_test_case: The test parameters to use.
        """
        # Arrange done in fixtures.
        # Act.
        numerator, denominator = unit_analysis.flatten(
            flatten_test_case.input_type)

        # Assert.
        assert numerator == flatten_test_case.expected_numerator
        assert denominator == flatten_test_case.expected_denominator

    def test_un_flatten(self, un_flatten_test_case: UnFlattenTest) -> None:
        """
        Tests that the un_flatten() function works.
        :param un_flatten_test_case: The test parameters to use.
        """
        # Arrange done in fixtures.
        # Act.
        compound = unit_analysis.un_flatten(
            un_flatten_test_case.numerator,
            un_flatten_test_case.denominator)

        # Assert.
        assert compound == un_flatten_test_case.expected_compound

    def test_simplify(self, simplify_test_case: SimplifyTest) -> None:
        """
        Tests that the simplify() function works.
        :param simplify_test_case: The test parameters to use.
        """
        # Arrange done in fixtures.
        # Act.
        simplified = unit_analysis.simplify(simplify_test_case.to_simplify)

        # Assert.
        assert simplified == simplify_test_case.simplified
