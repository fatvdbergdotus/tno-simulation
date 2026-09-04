from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable
import math
import matplotlib.pyplot as plt

# constants
RADIUS = 0.03 # meters
DENSITY = 7800 # kg/m^3
MASS = ( 4 / 3) * math.pi * RADIUS**3 * DENSITY  # kgs
INITAL_ANGLE = 45 # degrees

@dataclass
class State:
    t: float
    x: float
    y: float
    vx: float
    vy: float

class Force(ABC):
    @abstractmethod
    def calculate(self, t,state: State) -> tuple[float, float]:
        pass

class Gravity(Force):
    def __init__(self, mass: float, g: float = 9.81):
        self.mass = mass
        self.g = g

    def calculate(self, t, state: State):
        return 0.0, -self.mass * self.g

class Drag(Force):
    def __init__(self, drag_coefficient: float, air_density: float, frontal_area: float):
        self.drag_coefficient = drag_coefficient
        self.air_density = air_density
        self.frontal_area = frontal_area

    def calculate(self, t, state: State):
        v = (state.vx**2 + state.vy**2)**0.5
        if v == 0:
            return 0.0, 0.0
        drag_force_magnitude = 0.5 * self.drag_coefficient * self.air_density * self.frontal_area * v**2
        drag_force_x = -drag_force_magnitude * (state.vx / v)
        drag_force_y = -drag_force_magnitude * (state.vy / v)
        return drag_force_x, drag_force_y

class InitialThrust(Force):
    def __init__(self, force, duration, angle):
        self.force = force
        self.duration = duration
        self.angle = angle

    def calculate(self, t, state):
        if t < self.duration:
            return (self.force * math.cos(self.angle), self.force * math.sin(self.angle))
        return 0.0, 0.0

class Simulator(ABC):
    def __init__(self, forces: list[Force], dt: float, mass: float, stop_condition: Callable[[State], bool]):
        self.forces = forces
        self.dt = dt
        self.mass = mass
        self.stop_condition = stop_condition

    @abstractmethod
    def transition(self, state: State) -> State:
        pass

    def simulate(self, initial_state: State) -> list[State]:

        states = [initial_state]
        state = initial_state

        while not self.stop_condition(state):
            state = self.transition(state)
            states.append(state)

        return states

class ForwardEulerSimulator(Simulator):
    def transition(self, state: State) -> State:
        forces = [ force.calculate(state.t, state) for force in self.forces ]
        total_fx = sum(fx for fx, fy in forces)
        total_fy = sum(fy for fx, fy in forces)

        ax = total_fx / self.mass
        ay = total_fy / self.mass

        return State(state.t + self.dt, state.x + state.vx * self.dt, state.y + state.vy * self.dt,
                     state.vx + ax * self.dt, state.vy + ay * self.dt )

class ExplicitEulerSimulator(Simulator):
    def transition(self, state: State) -> State:
        forces = [ force.calculate(state.t, state) for force in self.forces ]
        total_fx = sum(fx for fx, fy in forces)
        total_fy = sum(fy for fx, fy in forces)

        ax = total_fx / self.mass
        ay = total_fy / self.mass

        new_vx = state.vx + ax * self.dt
        new_vy = state.vy + ay * self.dt

        return State(state.t + self.dt, state.x + new_vx * self.dt, state.y + new_vy * self.dt,
                     new_vx, new_vy )

# initial state for the simulation (starting at position (0, 0) and launching at 45 degrees with a speed of 200 m/s)
initial_state = State(0.0, 0.0, 0.0, 200 * math.cos(math.radians(INITAL_ANGLE)), 200 * math.sin(math.radians(INITAL_ANGLE)))

# initialize forces for the simulation
gravity = Gravity(mass= MASS, g=9.81)
drag = Drag(drag_coefficient=0.47, air_density=1.225, frontal_area=math.pi * RADIUS**2)
initial_thrust = InitialThrust(force=100.0, duration=5.0, angle = INITAL_ANGLE)

def main():
    # create a simulator instance
    simulator = ForwardEulerSimulator(forces=[gravity, drag], dt=0.0001, mass= MASS,
                                       stop_condition=lambda state: state.y < 0)
    states = simulator.simulate(initial_state)

    # create another simulator instance with thrust included
    simulator_with_thrust = ForwardEulerSimulator(forces=[gravity, drag, initial_thrust], dt=0.0001, mass= MASS,
                                       stop_condition=lambda state: state.y < 0)
    states_with_thrust = simulator_with_thrust.simulate(initial_state)

    # create a simulator instance using Explicit Euler method
    explicit_simulator = ExplicitEulerSimulator(forces=[gravity, drag], dt=0.0001, mass= MASS,
                                       stop_condition=lambda state: state.y < 0)
    explicit_states = explicit_simulator.simulate(initial_state)

    # create another simulator instance with thrust included using Explicit Euler method
    explicit_simulator_with_thrust = ExplicitEulerSimulator(forces=[gravity, drag, initial_thrust], dt=0.0001, mass= MASS,
                                       stop_condition=lambda state: state.y < 0)
    explicit_states_with_thrust = explicit_simulator_with_thrust.simulate(initial_state)

    # plot the trajectory
    plt.plot([state.x for state in states], [state.y for state in states])
    plt.plot([state.x for state in states_with_thrust], [state.y for state in states_with_thrust])
    plt.plot([state.x for state in explicit_states], [state.y for state in explicit_states])
    plt.plot([state.x for state in explicit_states_with_thrust], [state.y for state in explicit_states_with_thrust])
    plt.title("Projectile Motion with Drag and Thrust")
    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")
    plt.grid()
    plt.show()

main()

    