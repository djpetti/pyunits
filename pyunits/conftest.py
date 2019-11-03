from typing import Optional
import unittest.mock as mock

import numpy as np

import pytest

from pyunits.unit import Unit
from pyunits.unit_type import UnitType
from pyunits.types import Numeric

from pyunits.tests.testing_types import UnitFactory, UnitTypeFactory


@pytest.fixture
def unit_factory(unit_type_factory: UnitTypeFactory) -> UnitFactory:
    """
    A factory that creates a new (mock) Unit object. It takes a string name
    for the class. Two invocations with the same name will result in two
    instances of the same class.
    :param unit_type_factory: The factory for creating UnitTypes.
    :return: Function that returns a new Unit when called.
    """
    # Keep a registry of subclasses that we've dynamically created.
    names_to_subclasses = {}

    def _unit_factory_impl(class_name: str, raw: Numeric = 0.0,
                           unit_type: Optional[UnitType] = None) -> mock.Mock:
        """
        :param class_name: The name of the fake Unit subclass.
        :param raw: Optional raw numeric value to give the unit.
        :param unit_type: Optional specification of UnitType for created
        unit.
        """
        subclass = names_to_subclasses.get(class_name)
        if subclass is None:
            # Create the fake Unit subclass.
            subclass = type(class_name, (Unit,), {})
            names_to_subclasses[class_name] = subclass

        mock_unit = mock.Mock(spec=subclass)

        # Make sure the type is appropriate.
        if unit_type is None:
            unit_type = unit_type_factory(f"{class_name}_UnitType")
        mock_type_property = mock.PropertyMock(return_value=unit_type)
        type(mock_unit).type = mock_type_property
        # The spec should work reasonably for instances as well as the class
        # itself.
        type(mock_unit).type_class = mock_type_property

        # Set the raw value.
        raw = np.asarray(raw)
        mock_raw = mock.PropertyMock(return_value=raw)
        type(mock_unit).raw = mock_raw

        return mock_unit

    return _unit_factory_impl


@pytest.fixture
def unit_type_factory() -> UnitTypeFactory:
    """
    A factory that creates a new (mock) UnitType object. It takes a string
    name for the class.
    :return: Function that returns a new UnitType when called.
    """
    # Keep a registry of subclasses that we've dynamically created.
    names_to_subclasses = {}

    def _unit_type_factory_impl(class_name: str) -> mock.Mock:
        # Create the fake UnitType subclass.
        subclass = names_to_subclasses.get(class_name)
        if subclass is None:
            # Create the fake Unit subclass.
            subclass = type(class_name, (UnitType,), {})
            names_to_subclasses[class_name] = subclass

        return mock.Mock(spec=subclass)

    return _unit_type_factory_impl
