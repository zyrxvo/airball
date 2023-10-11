<div style="display: flex; justify-content: center; float: center;"><img src="https://github.com/zyrxvo/airball/raw/main/docs/img/airball.png" aspect=1 alt="AIRBALL Logo, a 3-body problem made to look like the letter A." title="AIRBALL Logo, a 3-body problem made to look like the letter A." height="128" width="128"></div>

# Welcome to AIRBALL

AIRBALL is a package for running and managing flybys using [REBOUND](https://github.com/hannorein/rebound). It is an extension to REBOUND, the popular N-body integrator.

*AIRBALL is currently in alpha testing. The APIs are subject to change without warning or backwards compatibility. Feedback and feature requests are very welcome.*

## Features

* Logic for handling the geometry of adding, running, and removing a flyby object in a REBOUND simulation.
* Stellar environments for generating and managing randomly generated stars from different stellar environments throughout the galaxy. 
* Initial mass functions for quickly generating samples from probability distributions. 
* Astropy.units integration to help you manage the mess of units and scales.
* Interactive examples for teaching and exploring AIRBALL’s functionality. 

## Installation

AIRBALL is installable via `pip` with one simple command

```zsh
pip install airball

```

The following packages should automatically be installed along with AIRBALL:

- `rebound`
- `numpy`
- `scipy`
- `joblib`
- `astropy`

## Contributors

* Garett Brown, University of Toronto, <garett.brown@mail.utoronto.ca>
* Hanno Rein, University of Toronto, <hanno@hanno-rein.de>
* Hasaan Mohsin, University of Toronto, <hasaan.mohsin@mail.utoronto.ca>
* Ryan Chao-Ming Lam, University of Waterloo, <ryan.lam1@uwaterloo.ca>
* Linda He, Ivy Shi, and others. 

AIRBALL is open source and you are invited to contribute to this project! 

## Acknowledgments

If you use this code or parts of this code for results presented in a scientific publication, we would greatly appreciate a citation.

## License

AIRBALL is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

AIRBALL is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with airball.  If not, see <http://www.gnu.org/licenses/>.
