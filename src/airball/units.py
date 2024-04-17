import astropy.units as _u
from astropy.units import *
twopi = 6.28318530717958623199592693708837032318115234375
yrtwopi = _u.def_unit('yrtwopi', _u.yr/twopi, format={'latex': r'(yr/2\pi)'})
yr2pi = _u.def_unit('yr2pi', _u.yr/twopi, format={'latex': r'(yr/2\pi)'})
stars = _u.def_unit('stars')
_u.add_enabled_units([yr2pi, yrtwopi])
_u.add_enabled_aliases({'msun': _u.solMass})


def isUnit(var):
    '''Determines if an object is an Astropy Quantity. Used for Stellar Environment initializations.'''
    return isinstance(var, (_u.core.IrreducibleUnit, _u.core.CompositeUnit, _u.Unit))

class UnitSet():
  '''
    A class for managing the units of another class.
    The default units are `[u.au, u.yr2pi, u.solMass, u.rad, u.km/u.s, u.stars/u.pc**3, u.stars]`.
    If a list of units is provided, the UnitSet will attempt to determine the units of length, time, mass, angle, velocity, density, and object. Thus, if a unit of length and a unit of time are given, but no unit of velocity, then UnitSet will calculate the unit of velocity from the given units of length and time. If no units are provided, the default units will be used.
    Two UnitSets are considered equal if the string representations of the units in each UnitSets are identical.

    Args:
      UNIT_SYSTEM (list): A list of Astropy Units describing the units of the system.

    Attributes:
      UNIT_SYSTEM (list): A list of Astropy Units describing the units of the system.
      units (dict): A dictionary of Astropy Units describing the units of the system. Can also access the dictionary from the object itself.
      length (astropy.units.Unit): The unit of length.
      time (astropy.units.Unit): The unit of time.
      mass (astropy.units.Unit): The unit of mass.
      angle (astropy.units.Unit): The unit of angle.
      velocity (astropy.units.Unit): The unit of velocity in length/time.
      density (astropy.units.Unit): The unit of density in object/length**3.
      object (airball.units.Unit): The unit of an object (such as a star).

    Example:
      ```python
      import airball
      import airball.units as u
      us1 = airball.tools.UnitSet([u.pc, u.Myr])
      us2 = airball.tools.UnitSet()
      print(us1 == us2) # False
      print(us1.velocity) # pc/Myr
      print(us2['velocity']) # au/yr2pi
      ```
  '''

  def __init__(self, UNIT_SYSTEM=[]) -> None:
    self._units = {'length': _u.au, 'time': _u.Myr, 'mass': _u.solMass, 'angle': _u.rad, 'velocity': _u.km/_u.s, 'object': stars, 'density': stars/_u.pc**3}
    self.UNIT_SYSTEM = UNIT_SYSTEM
    pass

  @property
  def units(self):
    return self._units

  @property
  def UNIT_SYSTEM(self):
    return self._UNIT_SYSTEM

  def __getitem__(self, key):
    if isinstance(key, str): return self.units[key]
    else: raise InvalidKeyException()

  def __setitem__(self, key, value):
    if isinstance(key, str):
      if isUnit(value): self.units[key] = value
      else: raise InvalidUnitException()
    else: raise InvalidKeyException()

  def __str__(self):
    s = '{'
    for key in self.units:
       s += f'{key}: {self.units[key].to_string()}, '
    s = s[:-2] + '}'
    return s

  def __repr__(self):
    s = '{'
    for key in self.units:
       s += f'{key}: {self.units[key].to_string()},\n'
    s = s[:-2] + '}'
    return s

  def __iter__(self):
    for k in self.units:
      yield self.units[k]

  def __eq__(self, other):
    # Determines if the string representations of the units in each UnitSets are identical.
    if isinstance(other, UnitSet):
        result = True
        for u1,u2 in zip(self, other):
            result = result and (u1.to_string() == u2.to_string())
        return result
    return NotImplemented

  def __hash__(self):
    # Overrides the default implementation
    data = []
    for d in sorted(self.__dict__.items()):
        try: data.append((d[0], tuple(d[1])))
        except: data.append(d)
    data = tuple(data)
    return hash(data)

  def values(self):
    return self.units.values()

  @property
  def length(self):
    return self._units['length']

  @length.setter
  def length(self, value):
    self.UNIT_SYSTEM = [value]

  @property
  def time(self):
    return self._units['time']

  @time.setter
  def time(self, value):
    self.UNIT_SYSTEM = [value]

  @property
  def mass(self):
    return self._units['mass']

  @mass.setter
  def mass(self, value):
    self.UNIT_SYSTEM = [value]

  @property
  def angle(self):
    return self._units['angle']

  @angle.setter
  def angle(self, value):
    self.UNIT_SYSTEM = [value]

  @property
  def velocity(self):
    return self._units['velocity']

  @velocity.setter
  def velocity(self, value):
    self.UNIT_SYSTEM = [value]

  @property
  def density(self):
    return self._units['density']

  @density.setter
  def density(self, value):
    self.UNIT_SYSTEM = [value]

  @property
  def object(self):
    return self._units['object']

  @object.setter
  def object(self, value):
    self.UNIT_SYSTEM = [value]

  @UNIT_SYSTEM.setter
  def UNIT_SYSTEM(self, UNIT_SYSTEM):
    if UNIT_SYSTEM != []:
      lengthUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(_u.m)]
      self._units['length'] = lengthUnit[0] if lengthUnit != [] else self._units['length']

      timeUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(_u.s)]
      self._units['time'] = timeUnit[0] if timeUnit != [] else self._units['time']

      velocityUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(_u.km/_u.s)]
      if velocityUnit == [] and timeUnit != [] and lengthUnit != []: velocityUnit = [lengthUnit[0]/timeUnit[0]]
      self._units['velocity'] = velocityUnit[0] if velocityUnit != [] else self._units['velocity']

      massUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(_u.kg)]
      self._units['mass'] = massUnit[0] if massUnit != [] else self._units['mass']

      angleUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(_u.rad)]
      self._units['angle'] = angleUnit[0] if angleUnit != [] else self._units['angle']

      objectUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(stars)]
      self._units['object'] = objectUnit[0] if objectUnit != [] else stars

      densityUnit = [this for this in UNIT_SYSTEM if this.is_equivalent(stars/_u.m**3)]
      densityUnit2 = [this for this in UNIT_SYSTEM if this.is_equivalent(1/_u.m**3)]
      if densityUnit == [] and densityUnit2 != []:
        densityUnit = [self._units['object'] * densityUnit2[0]]
      elif densityUnit == [] and objectUnit != [] and lengthUnit != []:
        densityUnit = [self._units['object']/self._units['length']**3]
      elif densityUnit == [] and densityUnit2 == [] and objectUnit != []:
         densityLength = [this for this in self._units['density'].bases if this.is_equivalent(_u.m)][0]
         densityUnit = [self._units['object']/densityLength**3]
      self._units['density'] = densityUnit[0] if densityUnit != [] else self._units['density']

    self._UNIT_SYSTEM = list(self._units.values())

############################################################
###################### Exceptions ##########################
############################################################

class InvalidKeyException(Exception):
  def __init__(self): super().__init__('Invalid key type.')

class InvalidUnitException(Exception):
  def __init__(self): super().__init__('Value is not a valid unit type.')
