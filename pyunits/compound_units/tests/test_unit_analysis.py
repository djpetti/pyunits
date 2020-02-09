from collections import OrderedDict as Od
from typing import Callable, Dict, FrozenSet, NamedTuple, Tuple, Union
import functools
import unittest.mock as mock

import pytest

from pyunits.compound_units.compound_unit import CompoundUnit
from pyunits.compound_units.compound_unit_type import CompoundUnitType
from pyunits.compound_units.div_unit import DivUnit
from pyunits.compound_units.mul_unit import MulUnit
from pyunits.compound_units.operations import Operation
from pyunits.compound_units import unit_analysis
from pyunits.tests.testing_types import UnitFactory, UnitTypeFactory
from pyunits.types import RequestType, CompoundTypeFactories, Numeric
from pyunits.unitless import UnitlessType
from pyunits.unit_interface import UnitInterface
from pyunits.unit_type import UnitType

# Type alias for a function that makes CompoundUnits.
CompoundUnitFactory = Callable[[Operation, UnitInterface, UnitInterface],
                               CompoundUnit]
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
# Type alias for a function that makes Unitless values.
UnitlessTypeFactory = Callable[[], UnitlessType]


class TestUnitAnalysis:
    """
    Tests for the unit_analysis module.
    """

    # Number of flatten() test cases we have.
    _NUM_FLATTEN_TESTS = 7
    # Number of simplify() test cases we have for UnitTypes.
    _NUM_SIMPLIFY_TYPE_TESTS = 18
    # Number of simplify() test cases we have for units.
    _NUM_SIMPLIFY_UNIT_TESTS = 7
    # Number of un_flatten() test cases we have.
    _NUM_UN_FLATTEN_TESTS = 8

    class FlattenTest(NamedTuple):
        """
        Represents a single test-case for the flatten() function.
        :param input_type: The input Unit or UnitType to try flattening.
        :param expected_numerator: The expected numerator set.
        :param expected_denominator: The expected denominator set.
        """
        input_type: unit_analysis.CompoundUnitOrType
        expected_numerator: Dict[unit_analysis.CompoundUnitOrType, int]
        expected_denominator: Dict[unit_analysis.CompoundUnitOrType, int]

    class UnFlattenTest(NamedTuple):
        """
        Represents a single test-case for the un_flatten() function.
        :param numerator: The numerator set to un-flatten.
        :param denominator: The denominator set to un-flatten.
        :param expected_compound: The expected CompoundUnitType that would
        result.
        :param type_factories: The compound unit type factories to use.
        """
        numerator: Dict[UnitType, int]
        denominator: Dict[UnitType, int]
        expected_compound: UnitType
        type_factories: CompoundTypeFactories

    class SimplifyTypeTest(NamedTuple):
        """
        Represents a single test-case for the simplify() function for UnitTypes.
        :param to_simplify: The CompoundUnitType to simplify.
        :param simplified: The expected simplified type.
        :param type_factories: The compound unit type factories to use.
        """
        to_simplify: CompoundUnitType
        simplified: CompoundUnitType
        type_factories: CompoundTypeFactories

    class SimplifyUnitTest(NamedTuple):
        """
        Represents a single test-case for the simplify() function for Units.
        :param to_simplify: The CompoundUnit to simplify.
        :param expected_raw: The expected raw value of the simplified unit.
        :param mock_type_factories: The compound unit type factories to use.
        :param mock_simplify_type: The mocked implementation of simplify() for
        UnitTypes.
        """
        to_simplify: CompoundUnit
        expected_raw: Numeric
        mock_type_factories: mock.Mock
        mock_simplify_type: mock.Mock

    class UnitOrTypeFactories(NamedTuple):
        """
        A pairing of a single Unit or UnitType factory with a corresponding
        CompoundUnit or CompoundUnitType factory.
        :param single: The factory for single units or types.
        :param compound: The factory for compound units or types.
        """
        single: Union[UnitFactory, UnitTypeFactory]
        compound: Union[CompoundUnitFactory, CompoundTypeFactory]

    @classmethod
    @pytest.fixture
    def compound_unit_factory(cls, compound_type_factory: CompoundTypeFactory
                              ) -> CompoundUnitFactory:
        """
        A factory that creates a new (mock) CompoundUnit object, when passed
        the operation being performed as well as the left and right sub-units.
        :param compound_type_factory: The factory for creating
        CompoundUnitTypes.
        :return: Function that creates a new CompoundUnit when called.
        """
        # Maps operations to their corresponding CompoundUnit subclasses.
        operations_to_classes = {Operation.MUL: MulUnit, Operation.DIV: DivUnit}

        def _compound_unit_factory_impl(operation: Operation,
                                        left_unit: UnitInterface,
                                        right_unit: UnitInterface
                                        ) -> CompoundUnit:
            mock_unit = mock.Mock(spec=operations_to_classes[operation])

            # Set the operation, left, and right properties.
            op_property = mock.PropertyMock(return_value=operation)
            type(mock_unit).operation = op_property

            left_property = mock.PropertyMock(return_value=left_unit)
            type(mock_unit).left = left_property

            right_property = mock.PropertyMock(return_value=right_unit)
            type(mock_unit).right = right_property

            # Make sure the type is appropriate.
            mock_type = compound_type_factory(operation, left_unit.type,
                                              right_unit.type)
            mock_type_property = mock.PropertyMock(return_value=mock_type)
            type(mock_unit).type = mock_type_property

            return mock_unit

        return _compound_unit_factory_impl

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

            # Set the operation, left, and right properties.
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
        :return: Function that creates fake CompoundUnitTypes.
        """
        def fake_compound_type_factory_impl(operation: Operation,
                                            left: UnitType,
                                            right: UnitType
                                            ) -> FakeCompoundType:
            arg_group = (left, right)
            if operation == Operation.MUL:
                # Argument order doesn't matter.
                arg_group = frozenset(arg_group)
            return operation, arg_group

        return fake_compound_type_factory_impl

    @classmethod
    @pytest.fixture
    def unitless_type_factory(cls) -> UnitlessTypeFactory:
        """
        Creates a factory for making UnitlessType instances. Also mocks the
        Unitless class for the test so that it uses the same factory.
        :return: A function that takes a raw value and returns a fake Unitless
        instance with that value.
        """
        # Because UnitlessType is used to wrap only Unitless, there is
        # just one canonical instance of UnitlessType, which we mock here.
        with mock.patch(unit_analysis.__name__ + ".Unitless") as mock_unitless:
            mock_unitless.mock_add_spec(UnitlessType)

            yield lambda: mock_unitless
            # Finalization done implicitly upon exit from context manager.

    @classmethod
    @pytest.fixture(params=range(2), ids=["unit", "type"])
    def unit_or_type_factories(cls, request: RequestType,
                               unit_factory: UnitFactory,
                               unit_type_factory: UnitTypeFactory,
                               compound_unit_factory: CompoundUnitFactory,
                               compound_type_factory: CompoundTypeFactory,
                               ) -> UnitOrTypeFactories:
        """
        Helper fixture that creates factories for both Units and UnitTypes. This
        is useful for testing functions that work with both.
        :param request: The PyTest request object to use for parametrization.
        :param unit_factory: The UnitFactory to use.
        :param unit_type_factory: The UnitTypeFactory to use.
        :param compound_unit_factory: The CompoundUnitFactory to use.
        :param compound_type_factory: The CompoundTypeFactory to use.
        :return: The selected factory.
        """
        factories = (cls.UnitOrTypeFactories(single=unit_factory,
                                             compound=compound_unit_factory),
                     cls.UnitOrTypeFactories(single=unit_type_factory,
                                             compound=compound_type_factory))
        return factories[request.param]

    @classmethod
    @pytest.fixture(params=range(_NUM_FLATTEN_TESTS))
    def flatten_test_case(cls, request: RequestType,
                          unit_or_type_factories: UnitOrTypeFactories
                          ) -> FlattenTest:
        """
        Creates a new FlattenTest object to try.
        :param request: The request to use for parametrization.
        :param unit_or_type_factories: The Unit or UnitType factories to use
        for creating inputs. (flatten() works on both Units and UnitTypes.)
        CompoundUnitTypes.
        :return: The FlattenTest to use.
        """
        test_index = request.param

        single1 = unit_or_type_factories.single("Single1")
        single2 = unit_or_type_factories.single("Single2")
        single3 = unit_or_type_factories.single("Single3")

        mul_factory = functools.partial(unit_or_type_factories.compound,
                                        Operation.MUL)
        div_factory = functools.partial(unit_or_type_factories.compound,
                                        Operation.DIV)

        # The list of tests that we want to perform.
        flatten_tests = [
            # When we try to flatten something already flat.
            cls.FlattenTest(input_type=single1,
                            expected_numerator={single1: 1},
                            expected_denominator={}),
            # When we try to flatten something simple.
            cls.FlattenTest(input_type=div_factory(single1, single2),
                            expected_numerator={single1: 1},
                            expected_denominator={single2: 1}),
            cls.FlattenTest(input_type=mul_factory(single1, single2),
                            expected_numerator={single1: 1, single2: 1},
                            expected_denominator={}),
            cls.FlattenTest(input_type=mul_factory(single1, single1),
                            expected_numerator={single1: 2},
                            expected_denominator={}),
            # When we try to flatten something nested.
            cls.FlattenTest(input_type=div_factory(mul_factory(single1,
                                                               single2),
                                                   mul_factory(single2,
                                                               single3)),
                            expected_numerator={single1: 1, single2: 1},
                            expected_denominator={single2: 1, single3: 1}),
            cls.FlattenTest(input_type=div_factory(mul_factory(single1,
                                                               single1),
                                                   div_factory(
                                                       div_factory(single2,
                                                                   single1),
                                                       mul_factory(single3,
                                                                   single3),
                                                   )),
                            expected_numerator={single1: 3, single3: 2},
                            expected_denominator={single2: 1}),
            cls.FlattenTest(input_type=mul_factory(single1,
                                                   div_factory(single2,
                                                               single3)),
                            expected_numerator={single1: 1, single2: 1},
                            expected_denominator={single3: 1}),
        ]

        assert len(flatten_tests) == cls._NUM_FLATTEN_TESTS
        return flatten_tests[test_index]

    @classmethod
    @pytest.fixture(params=range(_NUM_SIMPLIFY_TYPE_TESTS))
    def simplify_type_test_case(cls, request: RequestType,
                                unit_type_factory: UnitTypeFactory,
                                compound_type_factory: CompoundTypeFactory,
                                fake_compound_type_factory:
                                FakeCompoundTypeFactory,
                                unitless_type_factory: UnitlessTypeFactory,
                                ) -> SimplifyTypeTest:
        """
        Creates a new SimplifyTest object to try for simplifying types.
        :param request: The request to use for parametrization.
        :param unit_type_factory: The factory to use for creating UnitTypes.
        :param compound_type_factory: The factory to use for creating
        CompoundUnitTypes.
        :param fake_compound_type_factory: The factory to use for creating
        fake CompoundUnitTypes.
        :param unitless_type_factory: The factory to use for creating Unitless
        instances.
        :return: The SimplifyTest to use.
        """
        test_index = request.param

        type1 = unit_type_factory("UnitType1")
        type2 = unit_type_factory("UnitType2")
        type3 = unit_type_factory("UnitType3")
        type4 = unit_type_factory("UnitType4")

        # Other instances of the same types.
        type1_other = unit_type_factory("UnitType1")
        # Fake the standard_unit_class() method so it returns something
        # predictable.
        type1.standard_unit_class.return_value = type1
        type1_other.standard_unit_class.return_value = type1

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
        simplified4 = unitless_type_factory()
        simplified5 = real_div(unitless_type_factory(), type1)

        # All simplify tests should use the fake compound unit type factories.
        fake_type_factories = CompoundTypeFactories(mul=fake_mul, div=fake_div)
        simplify_test = functools.partial(cls.SimplifyTypeTest,
                                          type_factories=fake_type_factories)

        # The list of tests that we want to perform. We use ordered dict for
        # the attributes here because the iteration order can change the results
        # of un_flatten(), and we want them to be consistent.
        simplify_tests = [
           # When no simplification is necessary. In these cases, we should
           # just return the input.
           simplify_test(to_simplify=simplified1, simplified=simplified1),
           simplify_test(to_simplify=simplified2, simplified=simplified2),
           simplify_test(to_simplify=simplified3, simplified=simplified3),
           # Simple simplification cases.
           simplify_test(to_simplify=real_div(real_mul(type1, type2),
                                              real_mul(type2, type3)),
                         simplified=fake_div(type1, type3)),
           simplify_test(to_simplify=real_div(real_mul(
               real_mul(type1, type1),
               type2,
           ), type1),
                            simplified=fake_mul(type1, type2)),
           # Nested divisions.
           simplify_test(to_simplify=real_div(real_div(type1, type2),
                                              real_div(type3, type2)),
                         simplified=fake_div(type1, type3)),
           simplify_test(to_simplify=real_div(real_div(
               type2, real_mul(type1, type1)),
               real_div(type3, type1)),
                         simplified=fake_div(type2, fake_mul(type1, type3))),
           # Unitless values.
           simplify_test(to_simplify=simplified4, simplified=simplified4),
           simplify_test(to_simplify=real_mul(unitless_type_factory(),
                                              unitless_type_factory()),
                         simplified=unitless_type_factory()),
           simplify_test(to_simplify=real_mul(unitless_type_factory(), type1),
                         simplified=type1),
           simplify_test(to_simplify=simplified5, simplified=simplified5),
           simplify_test(to_simplify=real_div(type1, unitless_type_factory()),
                         simplified=type1),
           simplify_test(to_simplify=real_div(type1, real_mul(type1, type2)),
                         simplified=fake_div(unitless_type_factory(), type2)),
           # Cases with multiple instances of the same UnitType.
           simplify_test(to_simplify=real_mul(real_mul(type1, type2),
                                              real_mul(type1_other, type2)),
                         simplified=fake_mul(fake_mul(type1, type1),
                                             fake_mul(type2, type2))),
           simplify_test(to_simplify=real_div(real_mul(type1, type2),
                                              real_mul(type1_other, type3)),
                         simplified=fake_div(type2, type3)),
           simplify_test(to_simplify=real_div(real_mul(type1, type2),
                                              real_mul(type1_other, type2)),
                         simplified=unitless_type_factory()),
           simplify_test(to_simplify=real_div(real_mul(type1,
                                                       real_mul(type1, type1)),
                                              real_mul(type1, type1)),
                         simplified=type1),
           simplify_test(to_simplify=real_div(type1,
                                              real_mul(type1_other, type2)),
                         simplified=fake_div(unitless_type_factory(),
                                             type2))
        ]

        assert len(simplify_tests) == cls._NUM_SIMPLIFY_TYPE_TESTS
        return simplify_tests[test_index]

    @classmethod
    @pytest.fixture(params=range(_NUM_SIMPLIFY_UNIT_TESTS))
    def simplify_unit_test_case(cls, request: RequestType,
                                unit_factory: UnitFactory,
                                unit_type_factory: UnitTypeFactory,
                                compound_unit_factory: CompoundUnitFactory
                                ) -> SimplifyUnitTest:
        """
        Creates a new SimplifyUnitTest object to try.
        :param request: The request to use for parametrization.
        :param unit_factory: The factory to use for creating Units.
        :param unit_type_factory: The factory to use for creating UnitTypes.
        :param compound_unit_factory: The factory to use for creating
        CompoundUnits.
        :return: The SimplifyUnitTest to use.
        """
        test_index = request.param

        unit1 = unit_factory("Unit1", raw=2.0)
        unit2 = unit_factory("Unit2", raw=3.0)

        # Create two units of the same type.
        type1 = unit_type_factory("UnitType1")
        unit1_of_type1 = unit_factory("Type1Unit1", raw=5.0,
                                      unit_type_class=type1)
        unit2_of_type1 = unit_factory("Type1Unit2", raw=7.0,
                                      unit_type_class=type1)

        # Mock the standard values of these units, since they'll be used.
        unit1_of_type1.to_standard.return_value = unit_factory(
            "Type1Unit1Standard", raw=16.0)
        unit2_of_type1.to_standard.return_value = unit_factory(
            "Type1Unit2Standard", raw=8.0)

        mul_factory = functools.partial(compound_unit_factory, Operation.MUL)
        div_factory = functools.partial(compound_unit_factory, Operation.DIV)

        # Mock the version of simplify() for types. We simply do this by
        # registering an alternate version.
        mock_simplify_type = mock.Mock()
        unit_analysis.simplify.register(UnitType, mock_simplify_type)

        # Create a fake CompoundTypeFactories instance, which will never
        # actually be used.
        compound_type_factories = mock.Mock(spec=CompoundTypeFactories)
        simplify_test = functools.partial(
            cls.SimplifyUnitTest, mock_type_factories=compound_type_factories,
            mock_simplify_type=mock_simplify_type
        )

        simplify_tests = [
            # When no standardization is required.
            simplify_test(to_simplify=unit1, expected_raw=2.0),
            simplify_test(to_simplify=mul_factory(unit1, unit2),
                          expected_raw=6.0),
            simplify_test(to_simplify=div_factory(mul_factory(unit1, unit2),
                                                  unit2),
                          expected_raw=2.0),
            # When standardization is required.
            simplify_test(to_simplify=mul_factory(unit1_of_type1,
                                                  unit2_of_type1),
                          expected_raw=128.0),
            simplify_test(to_simplify=div_factory(unit1_of_type1,
                                                  unit2_of_type1),
                          expected_raw=2.0),
            simplify_test(to_simplify=mul_factory(div_factory(unit1,
                                                              unit1_of_type1),
                                                  div_factory(unit2_of_type1,
                                                              unit2)),
                          expected_raw=(1.0 / 3.0)),
            # When we have multiple units of the same class. (It should not
            # standardize.)
            simplify_test(to_simplify=mul_factory(unit1_of_type1,
                                                  unit1_of_type1),
                          expected_raw=25.0)
        ]

        assert len(simplify_tests) == cls._NUM_SIMPLIFY_UNIT_TESTS
        yield simplify_tests[test_index]

        # Un-mock the simplify function so it can be used again.
        unit_analysis.simplify.register(UnitType, unit_analysis.simplify_type)

    @classmethod
    @pytest.fixture(params=range(_NUM_UN_FLATTEN_TESTS))
    def un_flatten_test_case(cls, request: RequestType,
                             unit_type_factory: UnitTypeFactory,
                             unitless_type_factory: UnitlessTypeFactory,
                             fake_compound_type_factory: FakeCompoundTypeFactory
                             ) -> UnFlattenTest:
        """
        Creates a new UnFlattenTest object to try.
        :param request: The request to use for parametrization.
        :param unit_type_factory: The factory to use for creating UnitTypes.
        :param unitless_type_factory: The factory to use for creating Unitless
        instances.
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

        # All un-flatten tests should use the fake compound unit type factories.
        fake_type_factories = CompoundTypeFactories(mul=mul_factory,
                                                    div=div_factory)
        un_flatten_test = functools.partial(cls.UnFlattenTest,
                                            type_factories=fake_type_factories)

        # The list of tests that we want to perform. We use ordered dict for
        # the attributes here because the iteration order can change the results
        # of un_flatten(), and we want them to be consistent.
        un_flatten_tests = [
            # When we have no denominator.
            un_flatten_test(numerator=Od({type1: 1}), denominator={},
                            expected_compound=type1),
            un_flatten_test(numerator=Od({type1: 2}), denominator={},
                            expected_compound=mul_factory(type1, type1)),
            # When we do have a denominator.
            un_flatten_test(numerator=Od({type1: 1}),
                            denominator=Od({type2: 2}),
                            expected_compound=div_factory(
                                type1,
                                mul_factory(type2, type2))),
            un_flatten_test(numerator=Od({type1: 1, type2: 1}),
                            denominator=Od({type3: 2}),
                            expected_compound=div_factory(
                                mul_factory(type1, type2),
                                mul_factory(type3, type3),
                            )),
            # When we have no numerator.
            un_flatten_test(numerator={}, denominator={type1: 1},
                            expected_compound=div_factory(
                                unitless_type_factory(),
                                type1,
                            )),
            un_flatten_test(numerator={}, denominator={type1: 2, type2: 1},
                            expected_compound=div_factory(
                                unitless_type_factory(),
                                mul_factory(mul_factory(type1, type2),
                                            type1),
                            )),
            # When we have nested products.
            un_flatten_test(numerator=Od({type1: 2, type2: 2}),
                            denominator=Od({type3: 2, type4: 2}),
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
            un_flatten_test(numerator=Od({type1: 2, type2: 1}),
                            denominator=Od({type3: 1, type4: 2}),
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

        assert len(un_flatten_tests) == cls._NUM_UN_FLATTEN_TESTS
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
            un_flatten_test_case.denominator,
            un_flatten_test_case.type_factories,
        )

        # Assert.
        assert compound == un_flatten_test_case.expected_compound

    def test_simplify_type(self, simplify_type_test_case: SimplifyTypeTest
                           ) -> None:
        """
        Tests that the simplify() function works on UnitTypes.
        :param simplify_type_test_case: The test parameters to use.
        """
        # Arrange done in fixtures.
        # Act.
        simplified = unit_analysis.simplify(
            simplify_type_test_case.to_simplify,
            simplify_type_test_case.type_factories)

        # Assert.
        assert simplified == simplify_type_test_case.simplified

    def test_simplify_unit(self, simplify_unit_test_case: SimplifyUnitTest
                           ) -> None:
        """
        Tests that the simplify() function works on Units.
        :param simplify_unit_test_case: The test parameters to use.
        """
        # Arrange done in fixtures.
        # Act.
        simplified = unit_analysis.simplify(
            simplify_unit_test_case.to_simplify,
            simplify_unit_test_case.mock_type_factories)

        # Assert.
        # It should have created a new unit with the correct raw value.
        simplified_type = simplify_unit_test_case.mock_simplify_type\
            .return_value
        assert simplified == simplified_type.return_value
        simplified_type.assert_called_once_with(
            pytest.approx(simplify_unit_test_case.expected_raw))
