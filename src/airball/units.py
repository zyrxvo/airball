from astropy.units import *
twopi = 6.28318530717958623199592693708837032318115234375
yrtwopi = def_unit('yrtwopi', yr/twopi, format={'latex': r'(yr/2\pi)'})
yr2pi = def_unit('yr2pi', yr/twopi, format={'latex': r'(yr/2\pi)'})
stars = def_unit('stars')
add_enabled_units([yr2pi, yrtwopi])
add_enabled_aliases({'msun': solMass})