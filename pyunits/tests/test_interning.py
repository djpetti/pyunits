from typing import Any, NamedTuple, Tuple, Type

from pyunits.interning import Interned, InterningError

import pytest


class TestInterned:
    """
    Tests for the Interned class.
    """

    class InternedConfig(NamedTuple):
        """
        Encapsulates configuration for a test.
        :param subclass: The Interned subclass that we are testing with.
        """
        subclass: Type

    @classmethod
    @pytest.fixture
    def config(cls) -> InternedConfig:
        """
        Generates new configuration for a test.
        :return: The configuration that it generated.
        """
        # Clear the cache before every test.
        Interned.clear_interning_cache()

        # Create a new Interned subclass.
        class TestInternedClass(Interned):

            def _init_new(self, state: Any, extra: Any = None) -> None:
                """
                See superclass for documentation.
                :param state: Arbitrary state that will be saved in this
                :param extra: Extra value that will be ignored for interning
                purposes.
                instance.
                """
                self.__state = state
                self.__extra = extra

            @classmethod
            def _pre_hash(cls, state: Any, extra: Any = None) -> Tuple:
                """
                Allows us to ignore the "extra" argument when interning.
                See _init_new() for parameter documentation.
                :return: A tuple that does not include the "extra" argument.
                """
                return tuple([state])

            @property
            def state(self) -> Any:
                """
                :return: The state saved in this instance.
                """
                return self.__state

            @property
            def extra(self) -> Any:
                """
                :return: The value passed as the "extra" parameter.
                """
                return self.__extra

        return cls.InternedConfig(subclass=TestInternedClass)

    def test_get(self, config: InternedConfig) -> None:
        """
        Tests that get() works.
        :param config: The testing configuration.
        """
        # Arrange done in fixture.
        # Act.
        instance1 = config.subclass.get(1)
        instance2 = config.subclass.get(2)
        instance3 = config.subclass.get(1)

        # Assert.
        # The first two instances should be different.
        assert instance1 != instance2
        # The third should be the same as the first, because we passed the same
        # arguments to get().
        assert instance1 == instance3

        # The set states should all match up.
        assert instance1.state == 1
        assert instance2.state == 2
        assert instance3.state == 1

    def test_get_pre_hash(self, config: InternedConfig) -> None:
        """
        Tests that get() still works with a custom pre-hash function.
        :param config: The testing configuration.
        """
        # Arrange done in fixture.
        # Act.
        instance1 = config.subclass.get(1, extra="instance1")
        instance2 = config.subclass.get(2, extra="instance2")
        instance3 = config.subclass.get(1, extra="instance3")

        # Assert.
        # The first two instances should be different.
        assert instance1 != instance2
        # The third should be the same as the first, because we passed the same
        # arguments to get().
        assert instance1 == instance3

        # The set states should all match up.
        assert instance1.state == 1
        assert instance2.state == 2
        assert instance3.state == 1

        assert instance1.extra == "instance1"
        assert instance2.extra == "instance2"
        # The third instance should also have extra set to "instance1", because
        # it ignores this argument when interning.
        assert instance3.extra == "instance1"

    def test_clear_interning_cache(self, config: InternedConfig) -> None:
        """
        Tests that clear_interning_cache() works.
        :param config: The testing configuration.
        """
        # Arrange.
        # Create a new instance.
        instance1 = config.subclass.get(1)

        # Act.
        # Clear the cache.
        config.subclass.clear_interning_cache()
        # Get the same instance again.
        instance2 = config.subclass.get(2)

        # Assert.
        # It should have broken the interning pattern because we cleared the
        # cache in between.
        assert instance1 != instance2

    def test_constructor_disabled(self, config: InternedConfig) -> None:
        """
        Tests that it's smart enough not to let us call the constructor directly
        on interned classes.
        :param config: THe testing configuration.
        """
        # Arrange done in fixtures.
        # Act and assert.
        with pytest.raises(InterningError, match="use get()"):
            config.subclass(1)
