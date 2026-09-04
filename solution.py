from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class State:
    t: float
    x: float
    y: float
    vx: float
    vy: float

class Force(ABC):
    @abstractmethod
    def calculate(self, state: State) -> tuple[float, float]:
        pass

class Gravity(Force):
    def __init__(self, mass: float, g: float = 9.81):
        self.mass = mass
        self.g = g

    def calculate(self, state: State):
        return 0.0, -self.mass * self.g

class Friction(Force):
    def __init__(self, k: float):
        self.k = k

    def calculate(self, state: State):
        return -self.k * state.vx, -self.k * state.vy

class InitialThrust(Force):

    def __init__(self, force, duration):
        self.force = force
        self.duration = duration

    def __call__(self, t, state):
        if t < self.duration:
            return self.force
        return 0.0
