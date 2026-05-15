# Module airball.imf

"The initial mass function (IMF) is an empirical function that describes the initial distribution of masses for a population of stars during star formation" [(wikipedia)](https://en.wikipedia.org/wiki/Initial_mass_function).

`airball.IMF` is a class for implementing and managing IMFs. Given a probability distribution, `AIRBALL` will normalize and prepare the distribution for sampling in order to quickly generate random samples from it.

The following documentation was automatically generated from the docstrings.

::: airball.imf.IMF

## Available IMFs

### The MassFunction Protocol

The `airball.IMF` class is designed to take any callable function provided as a mass function for sampling. However, a protocol is defined for attaching units to custom mass functions to help maintain unit consistency and help prevent unintentional errors.

::: airball.imf.MassFunction

### Provided IMFs

::: airball.imf
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members_order: alphabetical
      filters:
        - "!^IMF$"
        - "!^MassFunction$"
