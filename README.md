# Installation

```bash
pip install python-pyunits
```

Here's something we've all done before:
```python
length1 = 42 # Meters
 
# ... 500 lines later

length2 = 12 # Inches
result = length1 + length2 # Oops
```

# PyUnits
*Use Units in your Python Code*

> **WARNING**: This code is still under development. Unless you are invigorated
> by the thought of an extended correspondence with me, I would recommend
> **against** its use for any serious *(read: paid)* project.

## Units are a pain.

And any time we're computing with some sort of physical quantities, we almost
certainly have to take them into account. Whenever I read (or write) this kind
of code, I am amazed at how many subtle bugs crop up due to missing or
incorrect unit conversions.

## WolframAlpha can do it, so why can't Python?

Why indeed? We're used to scientific software and even search engines auto-magically
handling unit conversions and dimensional analysis for us. Python is used for a
lot of scientific applications, but sadly, this feature isn't built-in to the
language.

Ladies and gentlemen, I give you PyUnits:

```python
from examples import example_units as eu

length1 = eu.Meters(42)
length2 = eu.Inches(12)

print(length1 + length2) # Prints "42.3048 m"
```

Okay. So maybe you're not turned on by unit conversions. I can understand that.
But guess what? PyUnits also understands dimensional analysis. In fact, it
understands it at least as well as [Randall Munroe](https://en.wikipedia.org/wiki/Randall_Munroe)
does:

![Dimensional Analysis](https://imgs.xkcd.com/comics/dimensional_analysis.png)

```python
from examples import example_units as eu
from pyunits.compound_units import Mul, Div

CubicMeters = Mul(eu.Meters, Mul(eu.Meters, eu.Meters))
CubicIn = Mul(eu.Inches, Mul(eu.Inches, eu.Inches))
Pascals = Div(eu.Joules, CubicMeters)
MilesPerIn3 = Div(eu.Miles, CubicIn)

plank_energy = eu.Joules(1.956E9)
core_pressure = Pascals(3.6E11)
# Technically, this is the Prius C gas mileage. It seems that Priuses
# (Priui?) have improved somewhat since the comic was published...
# Note also that we don't really have the ability to do mpg directly yet,
# so we do mi / in^3 and divide by 231. (1 gal = 231 in^3)
prius_mileage = MilesPerIn3(46 / 231)
channel_width = eu.Kilometers(33.8)

pi = plank_energy / core_pressure * prius_mileage / channel_width

print(pi) # Prints 3.14
```

"But wait", I hear you cry. "That's a totally contrived example!" And it may
well be. So let's try something a little more... practical:

![Fermirotica](https://imgs.xkcd.com/comics/fermirotica.png)

```python
import math

from examples import example_units as eu
from pyunits.compound_units import Mul

SquareMiles = Mul(eu.Miles, eu.Miles)
SquareMeters = Mul(eu.Meters, eu.Meters)

population_density = 18600 / SquareMiles(1.0)
sex_frequency = 80 / eu.Years(1.0)
sex_duration = eu.Minutes(30)

denom_value = math.pi * population_density * sex_frequency * sex_duration
fraction = 2.0 / denom_value

# Ensure result is in square meters.
fraction = SquareMeters(fraction)

# Square root has to be done outside of PyUnits, since we don't support that
# (yet).
sex_radius = math.sqrt(fraction.raw)
print(sex_radius) # Prints "139.33", the correct result for these parameters.
```

As you can see, PyUnits is suitable for Serious Scientific Work.

# Cookbook

I have a colleague who likes nothing better than copying and pasting code from
README files. He refers to this type of file as a "cookbook". Therefore, I will
be providing a similar facility in this section.

## Building a Unit Library

So far, all the examples in this document have used the example unit library
provided in this repository under `examples/example_units.py`. That's all well
and good, but if you're using PyUnits for real, the first thing you're probably
going to want to do is create your own unit library:

```python
from pyunits.unit import StandardUnit, Unit
from pyunits.unit_type import UnitType


class Length(UnitType):
    """
    Type for length units.
    """

@Length.decorate
class Meters(StandardUnit):
    """
    A meters unit.
    """

    @property
    def name(self) -> str:
        """
        See superclass for documentation.
        """
        return "m"


@Length.decorate
class Centimeters(Unit):
    """
    A centimeters unit.
    """

    def _from_standard(self, standard_value: StandardUnit) -> None:
        """
        See superclass for documentation.
        """
        # Convert from meters.
        self._set_raw(standard_value.raw * 100)

    def to_standard(self) -> Meters:
        """
        See superclass for documentation.
        """
        # Convert to meters.
        return Meters(self.raw / 100)

    @property
    def name(self) -> str:
        """
        See superclass for documentation.
        """
        return "cm"
```

Fundamentally, PyUnits has two types of objects that the user needs to be aware
of: `Unit` and `UnitType`. The former is easy: a `Unit` instance simply
represents a value with a specific unit. The second is a little harder to
explain.

Intuitively, some units can be trivially converted to each-other, eg. Meters
and Inches, and some can't be, eg. Meters and Seconds. The `UnitType` class
exists in order to allow PyUnits to understand these relationships. When using
PyUnits, every `Unit` subclass must be decorated with an appropriate `UnitType`
subclass. Two units that are marked with the same `UnitType` can be trivially
converted to each-other. Two units that aren't can't easily be converted, and
PyUnits with raise an error if you try to do so.

### Standard Units

For all the units of a particular `UnitType`, PyUnits expects the user to choose
a "standard unit". In practice, which unit you use as the standard one doesn't
really matter, as long as you can tell PyUnits how to convert from every other
unit to the standard one and vice-versa.

In practice, this is done by overriding two methods:
 - The `_from_standard()` method takes an instance of the standard unit and
   initializes this `Unit` instance appropriately with its converted value.
 - The `to_standard()` method returns an instance of the standard unit with
   an equivalent value to this one.
   
Once we have that set up, PyUnits can convert implicitly between all the units
of this `UnitType`:

```python
meters = Meters(10)
centimeters = Centimeters(meters)

print(centimeters)  # prints "1000 cm"
```

Which unit is considered to be the standard one is defined by which one
inherits from the `StandardUnit` class. (This is `Meters` in the above example.)

### Pretty-Printing

PyUnits has (currently limited) support for pretty-printing unit values. This
is utilized by overriding the `name` property, as seen in the example above.
This property should return a suffix that will be appended to the unit value
when printing.

### Numpy Integration

PyUnits can essentially be thought of as a wrapper around Numpy. That is
because `Unit` subclasses actually store and manipulate Numpy arrays internally:

```python
import numpy as np

from examples import example_units as eu

secs = eu.Seconds(np.array([1, 2, 3]))
print(secs)  # prints "[1 2 3] s"
```

You can access the raw Numpy value of a unit using the `raw` property:

```python
print(secs.raw)  # print "[1 2 3]"
```

## Dimensional Analysis

PyUnits is generally clever about unit operations:

```python
from examples import example_units as eu

meters = eu.Meters(2)
seconds = eu.Seconds(4)

print(meters / seconds)  # prints 0.5 m\n-\ns
```

As can be seen in the earlier examples, it will even go so far as to
auto-simplify the results of multiplication and division operations.

Also, compound unit types can be created manually:

```python
from examples import example_units as eu
from pyunits.compound_units import Mul, Div

Watts = Div(eu.Joules, eu.Seconds)
SquareMeters = Mul(eu.Meters, eu.Meters)

# These can then be used like normal units.
watts = Watts(10)
square_meters = SquareMeters(50)
```

### Unitless Values

PyUnits has the special concept of "Unitless" values, which, paradoxically,
can be used like a `Unit` in many cases. These crop up most often when doing
division.

```python
from examples import example_units as eu

meters1 = eu.Meters(10)
meters2 = eu.Meters(5)

print(meters1 / meters2)  # prints "2"
```

The result of this division is an instance of the class `Unitless`. This class
can also be used directly in order to represent concepts such as "inverse meters".

```python
from examples import example_units as eu
from pyunits.compound_units import Div
from pyunits.unitless import Unitless

inverse_meters = Div(Unitless, eu.Meters)
```

PyUnits will refuse to implicitly convert `Unitless` instances, however. If
you want to use this as a value of a unit, you have to explicitly take the raw
value:


```python
from examples import example_units as eu
from pyunits.unitless import Unitless

unitless = Unitless(5)
meters = eu.Meters(unitless)  # Error!

# The correct way...
meters = eu.Meters(unitless.raw)
```

This is an explicit design choice that was made to avoid cases where values
that did not have units could "magically" acquire them.


