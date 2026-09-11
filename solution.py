# Coding assignment TNO:PPAM
# To be found at: https://github.com/fatvdbergdotus/tno-simulation/
# (c) 2026 Freek van den Berg

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable

import math
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Projectile:
    """Physical properties and initial conditions of the projectile."""

    radius: float = 0.03                            # Projectile radius [m]
    density: float = 7800.0                         # Projectile density [kg/m^3]
    initial_velocity: float = 200.0                 # Initial speed [m/s]
    launch_angle_rad: float = math.radians(45.0)    # Launch angle [rad]


@dataclass(frozen=True)
class Environment:
    """Physical properties of the environment."""

    gravity: float = 9.81                # Gravitational acceleration [m/s^2]
    air_density: float = 1.225           # Air density [kg/m^3]
    drag_coefficient: float = 0.47       # Aerodynamic drag coefficient [-]


@dataclass
class State:
    """Represent the state of the projectile at a given point in time.

    Attributes:
        t: Time [s].
        x: Horizontal position [m].
        y: Vertical position [m].
        vx: Horizontal velocity [m/s].
        vy: Vertical velocity [m/s].
    """

    t: float
    x: float
    y: float
    vx: float
    vy: float


# Create the default projectile and environment.
PROJECTILE = Projectile()
ENVIRONMENT = Environment()


# ---------------------------------------------------------------------------
# Derived physical quantities and simulation settings
# ---------------------------------------------------------------------------

# Volume of a sphere:
#     V = 4/3 * pi * r^3
VOLUME: float = (
    (4 / 3) * math.pi * PROJECTILE.radius**3
)

# Mass follows from density times volume:
#     m = rho * V
MASS: float = VOLUME * PROJECTILE.density

# Frontal area of the spherical projectile:
#     A = pi * r^2
FRONTAL_AREA = math.pi * PROJECTILE.radius**2

# Time step used by the numerical integration.
DELTA_T: float = 0.0001  # [s]

# Stop the simulation once the projectile has passed below ground level.
STOP_CONDITION_HIT_GROUND: Callable[[State], bool] = (
    lambda state: state.y < 0
)


# ---------------------------------------------------------------------------
# Force models
# ---------------------------------------------------------------------------

class Force(ABC):
    """Abstract base class for forces acting on the projectile.

    Every force model must implement calculate(), which returns the
    horizontal and vertical force components.
    """

    @abstractmethod
    def calculate(
        self,
        state: State
    ) -> tuple[float, float]:
        """Calculate the force at the current projectile state.

        Args:
            state: Current state of the projectile.

        Returns:
            Tuple containing (Fx, Fy) in Newtons.
        """
        pass


class Gravity(Force):
    """Represent the gravitational force acting on the projectile."""

    def __init__(self, mass: float, g: float = 9.81):
        """Initialize the gravitational force.

        Args:
            mass: Projectile mass [kg].
            g: Gravitational acceleration [m/s^2].
        """
        self.mass = mass
        self.g = g

    def calculate(
        self,
        state: State
    ) -> tuple[float, float]:
        """Calculate the gravitational force.

        Gravity acts only in the negative y-direction.

        Args:
            state: Current projectile state.

        Returns:
            Tuple containing (Fx, Fy) in Newtons.
        """

        # There is no horizontal gravitational force.
        force_x = 0.0

        # Gravity acts downward.
        force_y = -self.mass * self.g

        return force_x, force_y


class Drag(Force):
    """Represent aerodynamic drag acting opposite to the velocity."""

    def __init__(
        self,
        drag_coefficient: float,
        air_density: float,
        frontal_area: float
    ):
        """Initialize the aerodynamic drag model.

        Args:
            drag_coefficient: Aerodynamic drag coefficient [-].
            air_density: Air density [kg/m^3].
            frontal_area: Projectile frontal area [m^2].
        """
        self.drag_coefficient = drag_coefficient
        self.air_density = air_density
        self.frontal_area = frontal_area

    def calculate(
        self,
        state: State
    ) -> tuple[float, float]:
        """Calculate the aerodynamic drag force.

        Drag magnitude is calculated using:

            F_D = 0.5 * C_D * rho * A * v^2

        The drag force always acts opposite to the velocity.

        Args:
            state: Current projectile state.

        Returns:
            Tuple containing (Fx, Fy) in Newtons.
        """

        # Calculate the projectile speed from its velocity components.
        #
        #     v = sqrt(vx^2 + vy^2)
        v = math.sqrt(state.vx**2 + state.vy**2)

        # If the projectile is stationary, there is no drag force.
        # This also prevents division by zero below.
        if v == 0:
            return 0.0, 0.0

        # Calculate the magnitude of the aerodynamic drag force.
        drag_force_magnitude = (
            0.5
            * self.drag_coefficient
            * self.air_density
            * self.frontal_area
            * v**2
        )

        # The unit vector in the direction of velocity is:
        #
        #     (vx / v, vy / v)
        #
        # Drag acts in the opposite direction, hence the minus signs.
        drag_force_x = (
            -drag_force_magnitude * (state.vx / v)
        )

        drag_force_y = (
            -drag_force_magnitude * (state.vy / v)
        )

        return drag_force_x, drag_force_y


class InitialThrust(Force):
    """Represent a constant thrust applied during the initial flight."""

    def __init__(
        self,
        force: float,
        duration: float,
        direction: float
    ):
        """Initialize the initial thrust.

        Args:
            force: Thrust magnitude [N].
            duration: Duration of the thrust [s].
            direction: Thrust direction [rad].
        """
        self.force = force
        self.duration = duration
        self.direction = direction

    def calculate(
        self,
        state: State
    ) -> tuple[float, float]:
        """Calculate the thrust force at the current time.

        Thrust is applied in a fixed direction while the current
        simulation time is less than the specified duration.

        Args:
            state: Current projectile state.

        Returns:
            Tuple containing (Fx, Fy) in Newtons.
        """

        # Apply thrust only during the initial part of the flight.
        if state.t < self.duration:

            # Resolve the thrust vector into horizontal and
            # vertical components.
            force_x = self.force * math.cos(self.direction)
            force_y = self.force * math.sin(self.direction)

            return force_x, force_y

        # After the thrust duration, the thrust is zero.
        return 0.0, 0.0


class CombinedForces(Force):
    """Combine several force models into one total force."""

    def __init__(self, forces):
        """Store the individual force models."""

        self.forces = forces

    def calculate(
        self,
        state: State
    ) -> tuple[float, float]:
        """Calculate the sum of all forces acting on the projectile."""

        # Calculate each individual force at the current state.
        forces = [
            force.calculate(state)
            for force in self.forces
        ]

        # Add all horizontal force components.
        total_fx = sum(
            fx for fx, _ in forces
        )

        # Add all vertical force components.
        total_fy = sum(
            fy for _, fy in forces
        )

        return total_fx, total_fy


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class Simulator(ABC):
    """Abstract base class for projectile simulators.

    The simulator contains the common simulation loop. Subclasses
    implement the numerical integration method in transition().
    """

    def __init__(
        self,
        force: Force,
        dt: float,
        mass: float,
        stop_condition: Callable[[State], bool]
    ):
        """Initialize the simulator.

        Args:
            force: Force model acting on the projectile.
            dt: Simulation time step [s].
            mass: Projectile mass [kg].
            stop_condition: Function determining when to stop.
        """

        self.force = force
        self.dt = dt
        self.mass = mass
        self.stop_condition = stop_condition

        # This list will contain every state generated by the simulation.
        self.states = []

    @abstractmethod
    def transition(
        self,
        state: State
    ) -> State:
        """Calculate the next state from the current state."""
        pass

    def get_description(self) -> str:
        """Return a short description of the simulator."""

        return (
            f"{type(self).__name__} "
            f"{self.dt} dt and {self.mass:.2f} mass"
        )

    def simulate(
        self,
        initial_state: State
    ) -> list[State]:
        """Run the simulation until the stop condition is satisfied.

        Args:
            initial_state: Initial state of the projectile.

        Returns:
            List containing all generated projectile states.
        """

        # Start the trajectory with the initial state.
        self.states = [initial_state]
        state = initial_state

        # Continue calculating states until the projectile hits the ground.
        while not self.stop_condition(state):

            # Calculate the next state using the selected
            # numerical integration method.
            state = self.transition(state)

            # Store the new state so it can later be analysed or plotted.
            self.states.append(state)

        return self.states

    def get_states(self) -> list[State]:
        """Return all states generated by the simulation."""

        return self.states

    def print_statistics(self) -> None:
        """Print final-state and estimated impact statistics."""

        print(40 * "-")
        print(self.get_description())

        # The final stored state is slightly below ground because
        # the simulation stops when y becomes negative.
        final_state = self.states[-1]

        print(f"Final time: {final_state.t:.2f} s")

        print(
            f"Final position: "
            f"({final_state.x:.2f}, {final_state.y:.2f}) m"
        )

        print(
            f"Final velocity: "
            f"({final_state.vx:.2f}, {final_state.vy:.2f}) m/s"
        )

        # At least two states are required to interpolate the exact
        # point at which the trajectory crosses y = 0.
        if len(self.states) >= 2:

            # These are the two states surrounding the ground crossing.
            last_state = self.states[-2]
            final_state = self.states[-1]

            # Calculate the fraction of the final time step at which
            # y reaches zero using linear interpolation.
            fraction = (
                -last_state.y
                / (final_state.y - last_state.y)
            )

            # Interpolate time at y = 0.
            t_hit = (
                last_state.t
                + (final_state.t - last_state.t) * fraction
            )

            # Interpolate horizontal position at y = 0.
            x_hit = (
                last_state.x
                + (final_state.x - last_state.x) * fraction
            )

            print(f"Time of impact: {t_hit:.2f} s")
            print(f"Distance traveled: {x_hit:.2f} m")


# ---------------------------------------------------------------------------
# Numerical integration methods
# ---------------------------------------------------------------------------

class ForwardEulerSimulator(Simulator):
    """Simulator using the standard Forward Euler method.

    Forward Euler uses the current velocity to update position.
    """

    def transition(
        self,
        state: State
    ) -> State:
        """Calculate the next state using Forward Euler."""

        # Calculate the total force acting on the projectile.
        fx, fy = self.force.calculate(state)

        # Newton's second law:
        #
        #     F = m * a
        #
        # Therefore:
        #
        #     a = F / m
        ax = fx / self.mass
        ay = fy / self.mass

        # Forward Euler uses the CURRENT velocity to calculate
        # the new position.
        new_x = state.x + state.vx * self.dt
        new_y = state.y + state.vy * self.dt

        # The acceleration is used to calculate the new velocity.
        new_vx = state.vx + ax * self.dt
        new_vy = state.vy + ay * self.dt

        return State(
            state.t + self.dt,
            new_x,
            new_y,
            new_vx,
            new_vy
        )


class ExplicitEulerSimulator(Simulator):
    """Simulator using semi-implicit (symplectic) Euler.

    Despite the class name, this implementation uses the
    semi-implicit/symplectic Euler method: velocity is updated first,
    and the new velocity is then used to update position.
    """

    def transition(
        self,
        state: State
    ) -> State:
        """Calculate the next state using semi-implicit Euler."""

        # Calculate the total force acting on the projectile.
        fx, fy = self.force.calculate(state)

        # Convert force to acceleration using Newton's second law.
        ax = fx / self.mass
        ay = fy / self.mass

        # First update the velocity.
        new_vx = state.vx + ax * self.dt
        new_vy = state.vy + ay * self.dt

        # Unlike Forward Euler, use the NEW velocity to calculate
        # the new position.
        new_x = state.x + new_vx * self.dt
        new_y = state.y + new_vy * self.dt

        return State(
            state.t + self.dt,
            new_x,
            new_y,
            new_vx,
            new_vy
        )


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

# The projectile starts at the origin:
#
#     x = 0 m
#     y = 0 m
#
# Its initial velocity is converted from magnitude and launch angle
# into horizontal and vertical velocity components.
initial_state = State(
    0.0,
    0.0,
    0.0,
    PROJECTILE.initial_velocity
    * math.cos(PROJECTILE.launch_angle_rad),
    PROJECTILE.initial_velocity
    * math.sin(PROJECTILE.launch_angle_rad)
)


# ---------------------------------------------------------------------------
# Force configuration
# ---------------------------------------------------------------------------

# Create the gravitational force.
gravity = Gravity(
    mass=MASS,
    g=ENVIRONMENT.gravity
)

# Create the aerodynamic drag force.
drag = Drag(
    drag_coefficient=ENVIRONMENT.drag_coefficient,
    air_density=ENVIRONMENT.air_density,
    frontal_area=FRONTAL_AREA
)

# Create an optional initial thrust force.
#
# The thrust acts for the first 5 seconds and points in the same
# direction as the initial launch angle.
initial_thrust = InitialThrust(
    force=100.0,
    duration=5.0,
    direction=PROJECTILE.launch_angle_rad
)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    descriptions: list[str],
    statess: list[list[State]]
) -> None:
    """Plot the trajectories produced by multiple simulations.

    Args:
        descriptions: Labels for the trajectories.
        statess: Lists of states produced by each simulator.
    """

    # Plot every trajectory using its corresponding description.
    for description, states in zip(descriptions, statess):

        # Extract x and y coordinates from all states.
        x_values = [
            state.x for state in states
        ]

        y_values = [
            state.y for state in states
        ]

        plt.plot(
            x_values,
            y_values,
            label=description
        )

    # Configure the plot.
    plt.title(
        "Projectile Motion with Gravity, Drag and Thrust"
    )

    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")

    plt.grid()

    # Add a legend so that the different simulation methods
    # can be distinguished.
    plt.legend(
        loc=2,
        prop={"size": 6}
    )

    # Save the generated figure.
    plt.savefig("solution.png")

    # Display the figure.
    plt.show()


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def engine() -> None:
    """Create, run and compare the projectile simulations."""

    # Four simulations are performed:
    #
    # 1. Forward Euler with gravity and drag.
    # 2. Forward Euler with gravity, drag and initial thrust.
    # 3. Semi-implicit Euler with gravity and drag.
    # 4. Semi-implicit Euler with gravity, drag and initial thrust.
    #
    # This makes it possible to compare both integration methods
    # and the effect of the initial thrust.

    simulators = [

        # Forward Euler without thrust.
        ForwardEulerSimulator(
            force=CombinedForces([
                gravity,
                drag
            ]),
            dt=DELTA_T,
            mass=MASS,
            stop_condition=STOP_CONDITION_HIT_GROUND
        ),

        # Forward Euler with initial thrust.
        ForwardEulerSimulator(
            force=CombinedForces([
                gravity,
                drag,
                initial_thrust
            ]),
            dt=DELTA_T,
            mass=MASS,
            stop_condition=STOP_CONDITION_HIT_GROUND
        ),

        # Semi-implicit Euler without thrust.
        ExplicitEulerSimulator(
            force=CombinedForces([
                gravity,
                drag
            ]),
            dt=DELTA_T,
            mass=MASS,
            stop_condition=STOP_CONDITION_HIT_GROUND
        ),

        # Semi-implicit Euler with initial thrust.
        ExplicitEulerSimulator(
            force=CombinedForces([
                gravity,
                drag,
                initial_thrust
            ]),
            dt=DELTA_T,
            mass=MASS,
            stop_condition=STOP_CONDITION_HIT_GROUND
        )
    ]

    # Store the trajectories and descriptions for plotting.
    statess = []
    descriptions = []

    # Run each simulation independently.
    for simulator in simulators:

        # Run the simulation from the common initial state.
        simulator.simulate(initial_state)

        # Store all generated states.
        statess.append(
            simulator.get_states()
        )

        # Store a description for the plot legend.
        descriptions.append(
            simulator.get_description()
        )

        # Print the simulation results.
        simulator.print_statistics()

    # Plot all four trajectories together.
    plot_results(
        descriptions,
        statess
    )


# ---------------------------------------------------------------------------
# Program entry point
# ---------------------------------------------------------------------------

# Only run the simulation automatically when this file is executed
# directly. Importing solution.py will not start the simulation.
if __name__ == "__main__":
    engine()