from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable
import math
import matplotlib.pyplot as plt


@dataclass
class State:
    """Represent the state of the projectile at a given point in time.

    Attributes:
        t: Time in seconds.
        x: Horizontal position in meters.
        y: Vertical position in meters.
        vx: Horizontal velocity in meters per second.
        vy: Vertical velocity in meters per second.
    """

    t: float
    x: float
    y: float
    vx: float
    vy: float


# Global constants
RADIUS: float = 0.03                                                    # m
DENSITY: float = 7800                                                   # kg/m^3
MASS: float = (4 / 3) * math.pi * RADIUS**3 * DENSITY                   # kg
FRONTAL_AREA = math.pi * RADIUS**2                                      # m^2
INITIAL_ANGLE: float = 45                                               # degrees
INITIAL_ANGLE_RAD: float = math.radians(INITIAL_ANGLE)                  # radians
INITIAL_VELOCITY: float = 200                                           # m/s
STOP_CONDITION: Callable[[State], bool] = lambda state: state.y < 0     # the simulations stop when the projectile hits the ground



class Force(ABC):
    """Abstract base class for forces acting on the projectile."""

    @abstractmethod
    def calculate(
        self,
        t: float,
        state: State
    ) -> tuple[float, float]:
        """Calculate the force components at the current state and time.

        Args:
            t: Current simulation time in seconds.
            state: Current state of the projectile.

        Returns:
            A tuple containing the force components (Fx, Fy) in Newtons.
        """
        pass


class Gravity(Force):
    """Represent the gravitational force acting on the projectile."""

    def __init__(self, mass: float, g: float = 9.81):
        """Initialize a gravitational force.

        Args:
            mass: Projectile mass in kilograms.
            g: Gravitational acceleration in meters per second squared.
        """
        self.mass = mass
        self.g = g

    def calculate(
        self,
        t: float,
        state: State
    ) -> tuple[float, float]:
        """Calculate the gravitational force.

        Gravity acts in the negative y-direction.

        Args:
            t: Current simulation time in seconds.
            state: Current projectile state.

        Returns:
            A tuple containing the gravitational force (Fx, Fy) in Newtons.
        """
        return 0.0, -self.mass * self.g


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
            drag_coefficient: Dimensionless aerodynamic drag coefficient.
            air_density: Density of the surrounding air in kg/m^3.
            frontal_area: Frontal area of the projectile in m^2.
        """
        self.drag_coefficient = drag_coefficient
        self.air_density = air_density
        self.frontal_area = frontal_area

    def calculate(
        self,
        t: float,
        state: State
    ) -> tuple[float, float]:
        """Calculate the aerodynamic drag force.

        The drag magnitude is calculated using:

            F_D = 0.5 * C_D * rho * A * v^2

        The resulting force acts in the direction opposite to
        the projectile velocity.

        Args:
            t: Current simulation time in seconds.
            state: Current projectile state.

        Returns:
            A tuple containing the drag force components (Fx, Fy)
            in Newtons.
        """
        v = (state.vx**2 + state.vy**2)**0.5 # calculate the velocity

        if v == 0:
            return 0.0, 0.0

        drag_force_magnitude = (
            0.5
            * self.drag_coefficient
            * self.air_density
            * self.frontal_area
            * v**2
        )

        drag_force_x = -drag_force_magnitude * (state.vx / v)
        drag_force_y = -drag_force_magnitude * (state.vy / v)

        return drag_force_x, drag_force_y


class InitialThrust(Force):
    """Represent a constant thrust applied during the initial flight."""

    def __init__(self, force: float, duration: float, direction: float):
        """Initialize the initial thrust.

        Args:
            force: Thrust magnitude in Newtons.
            duration: Duration of the thrust in seconds.
            direction: Direction of the thrust in radians.
        """
        self.force = force
        self.duration = duration
        self.direction = direction

    def calculate(
        self,
        t: float,
        state: State
    ) -> tuple[float, float]:
        """Calculate the thrust force at the current time.

        The thrust is applied at the initial launch angle while
        the current time is less than the specified duration.

        Args:
            t: Current simulation time in seconds.
            state: Current projectile state.

        Returns:
            A tuple containing the thrust force components (Fx, Fy)
            in Newtons. Returns (0, 0) after the thrust duration.
        """
        if t < self.duration:
            return (
                self.force * math.cos(self.direction),
                self.force * math.sin(self.direction)
            )

        return 0.0, 0.0


class Simulator(ABC):
    """Abstract base class for projectile simulators.

    The simulator maintains a sequence of projectile states and
    repeatedly applies a transition until the stop condition is met.
    """

    def __init__(
        self,
        forces: list[Force],
        dt: float,
        mass: float,
        stop_condition: Callable[[State], bool]
    ):
        """Initialize the simulator.

        Args:
            forces: List of forces acting on the projectile.
            dt: Simulation time step in seconds.
            mass: Projectile mass in kilograms.
            stop_condition: Function that determines when simulation stops.
        """
        self.forces = forces
        self.dt = dt
        self.mass = mass
        self.stop_condition = stop_condition
        self.states = []

    @abstractmethod
    def transition(self, state: State) -> State:
        """Calculate the next state from the current state.

        Args:
            state: Current projectile state.

        Returns:
            The next projectile state.
        """
        pass

    def get_description(self) -> str:
        """Return a human-readable description of the simulator.

        Returns:
            A string containing the simulator type, forces, time step,
            and projectile mass.
        """
        forces = ", ".join(
            type(force).__name__ for force in self.forces
        )

        return (
            f"{type(self).__name__} "
            f"with forces {forces}, "
            f"{self.dt} dt and {self.mass:.2f} mass"
        )

    def simulate(self, initial_state: State) -> list[State]:
        """Run the simulation until the stop condition is satisfied.

        Args:
            initial_state: Initial state of the projectile.

        Returns:
            A list containing all states generated during the simulation.
        """
        self.states = [initial_state]
        state = initial_state

        while not self.stop_condition(state):
            state = self.transition(state)
            self.states.append(state)

        return self.states

    def get_states(self) -> list[State]:
        """Return all states generated by the simulation.

        Returns:
            The sequence of projectile states.
        """
        return self.states

    def print_statistics(self) -> None:
        """Print the final state and estimated impact statistics."""
        print(40 * "-")
        print(self.get_description())

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

        # Linear interpolation to estimate the time and position
        # at which the projectile reaches y = 0.
        if len(self.states) >= 2:
            last_state = self.states[-2]
            final_state = self.states[-1]

            fraction = (
                -last_state.y
                / (final_state.y - last_state.y)
            )

            t_hit = (
                last_state.t
                + (final_state.t - last_state.t) * fraction
            )

            x_hit = (
                last_state.x
                + (final_state.x - last_state.x) * fraction
            )

            print(f"Time of impact: {t_hit:.2f} s")
            print(f"Distance traveled: {x_hit:.2f} m")


class ForwardEulerSimulator(Simulator):
    """Simulator using the standard Forward Euler integration method."""

    def transition(self, state: State) -> State:
        """Calculate the next state using Forward Euler integration.

        Forces are evaluated using the current state. The resulting
        acceleration is then used to update velocity and position.

        Args:
            state: Current projectile state.

        Returns:
            The next projectile state.
        """
        forces = [
            force.calculate(state.t, state)
            for force in self.forces
        ]

        total_fx = sum(fx for fx, fy in forces)
        total_fy = sum(fy for fx, fy in forces)

        ax = total_fx / self.mass
        ay = total_fy / self.mass

        return State(
            state.t + self.dt,
            state.x + state.vx * self.dt,
            state.y + state.vy * self.dt,
            state.vx + ax * self.dt,
            state.vy + ay * self.dt
        )


class ExplicitEulerSimulator(Simulator):
    """Simulator using semi-implicit (symplectic) Euler integration.

    Velocity is updated before position, so the newly calculated
    velocity is used to update the position.
    """

    def transition(self, state: State) -> State:
        """Calculate the next state using semi-implicit Euler.

        Args:
            state: Current projectile state.

        Returns:
            The next projectile state.
        """
        forces = [
            force.calculate(state.t, state)
            for force in self.forces
        ]

        total_fx = sum(fx for fx, _ in forces)
        total_fy = sum(fy for _, fy in forces)

        ax = total_fx / self.mass
        ay = total_fy / self.mass

        new_vx = state.vx + ax * self.dt
        new_vy = state.vy + ay * self.dt

        return State(
            state.t + self.dt,
            state.x + new_vx * self.dt,
            state.y + new_vy * self.dt,
            new_vx,
            new_vy
        )




# Initial state:
# Projectile starts at (0, 0) with a velocity of 200 m/s
# at an angle of 45 degrees.
initial_state = State(
    0.0,
    0.0,
    0.0,
    INITIAL_VELOCITY * math.cos(INITIAL_ANGLE_RAD),
    INITIAL_VELOCITY * math.sin(INITIAL_ANGLE_RAD)
)


# Initialize forces.
gravity = Gravity(mass=MASS, g=9.81)

drag = Drag(
    drag_coefficient=0.47,
    air_density=1.225,
    frontal_area=FRONTAL_AREA
)

initial_thrust = InitialThrust(
    force=100.0,
    duration=5.0,
    direction=INITIAL_ANGLE_RAD
)


def plot_results(
    descriptions: list[str],
    statess: list[list[State]]
) -> None:
    """Plot the trajectories of multiple simulations.

    Args:
        descriptions: Labels used in the plot legend.
        statess: Lists of states produced by each simulator.
    """
    for description, states in zip(descriptions, statess):
        plt.plot(
            [state.x for state in states],
            [state.y for state in states],
            label=description
        )

    plt.title("Projectile Motion with Gravity, Drag and Thrust")
    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")
    plt.grid()
    plt.legend()
    plt.show()


def engine() -> None:
    """Create, run, and compare the projectile simulations."""
    simulators = [
        ForwardEulerSimulator(
            forces=[gravity, drag],
            dt=0.0001,
            mass=MASS,
            stop_condition=STOP_CONDITION
        ),

        ForwardEulerSimulator(
            forces=[gravity, drag, initial_thrust],
            dt=0.0001,
            mass=MASS,
            stop_condition=STOP_CONDITION
        ),

        ExplicitEulerSimulator(
            forces=[gravity, drag],
            dt=0.0001,
            mass=MASS,
            stop_condition=STOP_CONDITION
        ),

        ExplicitEulerSimulator(
            forces=[gravity, drag, initial_thrust],
            dt=0.0001,
            mass=MASS,
            stop_condition=STOP_CONDITION
        )
    ]

    statess = []
    descriptions = []

    for simulator in simulators:
        simulator.simulate(initial_state)

        statess.append(simulator.get_states())
        descriptions.append(simulator.get_description())

        simulator.print_statistics()

    plot_results(descriptions, statess)


if __name__ == "__main__":
    engine()