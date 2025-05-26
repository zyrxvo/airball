# AIRBALL API
The API documentation is automatically generated from the docstrings.

## Submodules
`AIRBALL` contains several submodules specific to certain tasks or situations.

* [airball.core](core.md): functions and logic for running flybys, multiple flybys, or flybys in parallel.
* [airball.stars](stars.md): the `Star` and `Stars` classes for managing and manipulating the objects that will be flying by your `REBOUND` simulations.
* [airball.environments](environments.md): the `StellarEnvironment` class and subclasses for managing and generating random stellar flybys of particular types, (`OpenCluster`, `LocalNeighborhood`, `GlobularCluster`, etc.)
* [airball.imf](imf.md): the `IMF` class which manages the initial mass function of the stellar environments, or a custom IMF of your design.
* [airball.analytic](analytic.md): analytic functions for comparison to n-body results to assist in verification of n-body results.
* [airball.tools](tools.md): miscellaneous tools for plotting, analyzing, or supporting the functionality of `AIRBALL`.
* [airball.units](units.md): a small extension of `astropy.units` with the addition of units such as `yr2pi` ($\mathrm{yr}/2\pi$) and generic `stars`.
