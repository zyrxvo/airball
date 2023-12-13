# Module airball.core

This module contains the core functionality of `AIRBALL`. It handles the logic and geometry of adding, integrating, and removing a flyby object in a `REBOUND` simulation.

There are two main methods for integrating a flyby. The standard method which uses `IAS15` for the entire integration and a hybrid method, which switches from `WHCKL` to `IAS15` when the flyby star comes within a specified distance to the outermost object in the simulation.

!!! Info
    For most situations, we suggest using `IAS15` for the entire integration. For a 3-body system (Sun, Neptune, and flyby star), a typical flyby interaction often takes less than a second of wall-time to integrate. If you're finding that the integration is 'hanging', try setting `sim.ri_ias15.adaptive_mode = 2`. When the distance between the objects is very large, `IAS15` may struggle to converge at each timestep so it defaults to the minimum step size.

The following documentation was automatically generated from the docstrings.


::: airball.core.flyby
::: airball.core.flybys

::: airball.core.hybrid_flyby
::: airball.core.hybrid_flybys

::: airball.core.successive_flybys
::: airball.core.concurrent_flybys

::: airball.core.add_star_to_sim
::: airball.core.remove_star_from_sim

