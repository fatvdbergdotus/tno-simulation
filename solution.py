from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable
import math
import matplotlib.pyplot as plt

# global constants
RADIUS = 0.03 # meters
DENSITY = 7800 # kg/m^3
MASS = ( 4 / 3) * math.pi * RADIUS**3 * DENSITY  # kgs
INITAL_ANGLE = 45 # degrees
INITAL_ANGLE_RAD = math.radians(INITAL_ANGLE) # radians
INITIAL_VELOCITY = 200 # m/s

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
    def __init__(self, force, duration):
        self.force = force
        self.duration = duration

    def calculate(self, t, state):
        if t < self.duration:
            return (self.force * math.cos(INITAL_ANGLE_RAD), self.force * math.sin(INITAL_ANGLE_RAD))
        return 0.0, 0.0

class Simulator(ABC):
    def __init__(self, forces: list[Force], dt: float, mass: float, stop_condition: Callable[[State], bool]):
        self.forces = forces
        self.dt = dt
        self.mass = mass
        self.stop_condition = stop_condition
        self.states = []

    @abstractmethod
    def transition(self, state: State) -> State:
        pass

    def get_description(self) -> str:
        return f"{type(self).__name__} with forces {[type(force).__name__ for force in self.forces]}, {self.dt} dt and {self.mass:.2f} mass"

    def simulate(self, initial_state: State) -> list[State]:
        self.states = [initial_state]
        state = initial_state

        while not self.stop_condition(state):
            state = self.transition(state)
            self.states.append(state)

    def get_states(self):
        return self.states

    def print_statistics(self):
        print (20*"-")
        print (self.get_description())
        final_state = self.states[-1]
        print(f"Final time: {final_state.t:.2f} s")
        print(f"Final position: ({final_state.x:.2f}, {final_state.y:.2f}) m")
        print(f"Final velocity: ({final_state.vx:.2f}, {final_state.vy:.2f}) m/s")

        # linear interpolation to find the time when the projectile hits the ground (y=0)
        if len(self.states) >= 2:
            last_state = self.states[-2]
            final_state = self.states[-1]
            t_hit = last_state.t + (final_state.t - last_state.t) * (-last_state.y) / (final_state.y - last_state.y)
            x_hit = last_state.x + (final_state.x - last_state.x) * (-last_state.y) / (final_state.y - last_state.y)
            print(f"Time of impact: {t_hit:.2f} s")
            print(f"Distance traveled: {x_hit:.2f} m")    

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
initial_state = State(0.0, 0.0, 0.0, INITIAL_VELOCITY * math.cos(INITAL_ANGLE_RAD), INITIAL_VELOCITY * math.sin(INITAL_ANGLE_RAD))

# initialize forces for the simulation
gravity = Gravity(mass= MASS, g=9.81)
drag = Drag(drag_coefficient=0.47, air_density=1.225, frontal_area=math.pi * RADIUS**2)
initial_thrust = InitialThrust(force=100.0, duration=5.0)

def plot_results(descriptions, statess):
    # plot the trajectory
    for description, states in zip(descriptions, statess):
        plt.plot([state.x for state in states], [state.y for state in states], label=description)
    plt.title("Projectile Motion with Drag and Thrust")
    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")
    plt.grid()
    plt.legend()
    plt.show()


def engine():
    simulators = [
        ForwardEulerSimulator(forces=[gravity, drag], dt=0.0001, mass= MASS, stop_condition=lambda state: state.y < 0),
        ForwardEulerSimulator(forces=[gravity, drag, initial_thrust], dt=0.0001, mass= MASS, stop_condition=lambda state: state.y < 0),
        ExplicitEulerSimulator(forces=[gravity, drag], dt=0.0001, mass= MASS, stop_condition=lambda state: state.y < 0),
        ExplicitEulerSimulator(forces=[gravity, drag, initial_thrust], dt=0.0001, mass= MASS, stop_condition=lambda state: state.y < 0)
    ]
    statess=[]
    descriptions=[]
    for simulator in simulators:
        simulator.simulate(initial_state)
        statess.append(simulator.get_states())
        descriptions.append(simulator.get_description())
        simulator.print_statistics()

    plot_results(descriptions, statess)
engine()

    