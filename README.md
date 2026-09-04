# TNO Projectile Trajectory Simulator

Python implementation of a projectile trajectory simulation.

The simulator models a spherical projectile launched at an angle and calculates
its trajectory under the influence of different forces.

## Features

- Projectile state represented using a `State` dataclass
- Abstract `Force` interface
- Gravity force
- Aerodynamic drag force
- Initial thrust force
- Abstract `Simulator` class
- Forward Euler integration
- Explicit/Symplectic Euler integration
- Stop criterion using a lambda function
- Trajectory plotting using Matplotlib
- Impact time and distance calculation
- Comparison of multiple simulation configurations

## Requirements

Python 3.10 or newer.

Install Matplotlib:

```bash
pip install matplotlib
