# Module airball.units

`airball.units` extends `astropy.units` with a `yr2pi` quantity such that `yr2pi`$= \mathrm{yr}/(2\pi)$. This is the default time unit of `REBOUND` when the central mass object is one solar mass [$M_\odot$], the unit distance is one astronomical unit [AU], and the value of Newton's gravitational constant is 1.

`airball.units` also adds a `stars` quantity for keeping track of units of stars when doing encounter rates, density, and other calculations related to [`StellarEnvironments`](../environments/#airball.environments.StellarEnvironment).

Simply use `airball.units` the way you would use any `astropy.units`, but enjoy the additional unit of time.

!!! example "Example"
    ```python
    import rebound
    import airball.units as u
    sim = rebound.Simulation()
    sim.add(m=1)
    sim.add(m=5e-5, a=30, e=0.01)
    vel = (sim.particles[1].v * u.au/u.yr2pi).to(u.km/u.s) # Get velocity in km/s
    ```
