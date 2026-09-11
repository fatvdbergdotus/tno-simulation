from solution import *
import unittest

class TestConstantsAndProperties(unittest.TestCase):
    def test_projectile_properties(self):
        self.assertEqual(PROJECTILE.radius, 0.03)
        self.assertEqual(PROJECTILE.density, 7800.0)
        self.assertEqual(PROJECTILE.initial_velocity, 200.0)
        self.assertEqual(PROJECTILE.launch_angle_rad, math.radians(45.0))

    def test_enviroment_properties(self):
        self.assertEqual(ENVIRONMENT.gravity, 9.81)
        self.assertEqual(ENVIRONMENT.air_density, 1.225)
        self.assertEqual(ENVIRONMENT.drag_coefficient, 0.47)

    def test_constants(self):
        self.assertAlmostEqual(VOLUME, (4 / 3) * math.pi * PROJECTILE.radius**3)
        self.assertAlmostEqual(MASS, VOLUME * PROJECTILE.density)
        self.assertAlmostEqual(FRONTAL_AREA, math.pi * PROJECTILE.radius**2)
        self.assertAlmostEqual(DELTA_T, 0.0001)

class TestForces(unittest.TestCase):
    def test_gravity_force(self):
        gravity = Gravity(MASS, ENVIRONMENT.gravity)
        state = State(t=0, x=0, y=0, vx=0, vy=0)
        fx, fy = gravity.calculate(state)
        self.assertAlmostEqual(fx, 0.0)
        self.assertAlmostEqual(fy, -MASS * ENVIRONMENT.gravity)

    def test_drag_force(self):
        drag = Drag(ENVIRONMENT.drag_coefficient, ENVIRONMENT.air_density, FRONTAL_AREA)
        state = State(t=0, x=0, y=0, vx=10, vy=0)
        fx, fy = drag.calculate(state)
        expected_fx = -0.5 * ENVIRONMENT.drag_coefficient * ENVIRONMENT.air_density * FRONTAL_AREA * 10**2
        self.assertAlmostEqual(fx, expected_fx)
        self.assertAlmostEqual(fy, 0.0)

    def test_initial_thrust_force(self):
        thrust = InitialThrust(force=100.0, duration=2.0, direction=math.radians(45.0))
        state = State(t=0, x=0, y=0, vx=0, vy=0)
        fx, fy = thrust.calculate(state)
        expected_fx = 100.0 * math.cos(math.radians(45.0))
        expected_fy = 100.0 * math.sin(math.radians(45.0))
        self.assertAlmostEqual(fx, expected_fx)
        self.assertAlmostEqual(fy, expected_fy)

    def test_initial_thrust_force_after_duration(self):
        thrust = InitialThrust(force=100.0, duration=2.0, direction=math.radians(45.0))
        state = State(t=3, x=0, y=0, vx=0, vy=0)  # Time is after the thrust duration
        fx, fy = thrust.calculate(state)
        self.assertAlmostEqual(fx, 0.0)
        self.assertAlmostEqual(fy, 0.0)

    def test_combined_forces(self):
        gravity = Gravity(MASS, ENVIRONMENT.gravity)
        drag = Drag(ENVIRONMENT.drag_coefficient, ENVIRONMENT.air_density, FRONTAL_AREA)
        thrust = InitialThrust(force=100.0, duration=2.0, direction=math.radians(45.0))
        state = State(t=1, x=0, y=0, vx=10, vy=0)  # Time is within the thrust duration
        fx_gravity, fy_gravity = gravity.calculate(state)
        fx_drag, fy_drag = drag.calculate(state)
        fx_thrust, fy_thrust = thrust.calculate(state)

        fx_total = fx_gravity + fx_drag + fx_thrust
        fy_total = fy_gravity + fy_drag + fy_thrust

        expected_fx_thrust = 100.0 * math.cos(math.radians(45.0))
        expected_fy_thrust = 100.0 * math.sin(math.radians(45.0))
        expected_fx_total = 0.0 + (-0.5 * ENVIRONMENT.drag_coefficient * ENVIRONMENT.air_density * FRONTAL_AREA * 10**2) + expected_fx_thrust
        expected_fy_total = -MASS * ENVIRONMENT.gravity + 0.0 + expected_fy_thrust

        self.assertAlmostEqual(fx_total, expected_fx_total)
        self.assertAlmostEqual(fy_total, expected_fy_total)

    def test_combined_forces_after_thrust_duration(self):
        gravity = Gravity(MASS, ENVIRONMENT.gravity)
        drag = Drag(ENVIRONMENT.drag_coefficient, ENVIRONMENT.air_density, FRONTAL_AREA)
        thrust = InitialThrust(force=100.0, duration=2.0, direction=math.radians(45.0))
        state = State(t=3, x=0, y=0, vx=10, vy=0)  # Time is after the thrust duration
        fx_gravity, fy_gravity = gravity.calculate(state)
        fx_drag, fy_drag = drag.calculate(state)
        fx_thrust, fy_thrust = thrust.calculate(state)

        fx_total = fx_gravity + fx_drag + fx_thrust
        fy_total = fy_gravity + fy_drag + fy_thrust

        expected_fx_thrust = 0.0  # Thrust has ended
        expected_fy_thrust = 0.0
        expected_fx_total = 0.0 + (-0.5 * ENVIRONMENT.drag_coefficient * ENVIRONMENT.air_density * FRONTAL_AREA * 10**2) + expected_fx_thrust
        expected_fy_total = -MASS * ENVIRONMENT.gravity + 0.0 + expected_fy_thrust

        self.assertAlmostEqual(fx_total, expected_fx_total)
        self.assertAlmostEqual(fy_total, expected_fy_total)

class TestSimulation(unittest.TestCase):
    def test_simulation_step(self):
        gravity = Gravity(MASS, ENVIRONMENT.gravity)
        drag = Drag(ENVIRONMENT.drag_coefficient, ENVIRONMENT.air_density, FRONTAL_AREA)
        thrust = InitialThrust(force=100.0, duration=2.0, direction=math.radians(45.0))
        state = State(t=1, x=0, y=0, vx=10, vy=0)  # Initial state at t=1
        fx_gravity, fy_gravity = gravity.calculate(state)
        fx_drag, fy_drag = drag.calculate(state)
        fx_thrust, fy_thrust = thrust.calculate(state)

        fx_total = fx_gravity + fx_drag + fx_thrust
        fy_total = fy_gravity + fy_drag + fy_thrust

        expected_fx_thrust = 100.0 * math.cos(math.radians(45.0))
        expected_fy_thrust = 100.0 * math.sin(math.radians(45.0))
        expected_fx_total = 0.0 + (-0.5 * ENVIRONMENT.drag_coefficient * ENVIRONMENT.air_density * FRONTAL_AREA * 10**2) + expected_fx_thrust
        expected_fy_total = -MASS * ENVIRONMENT.gravity + 0.0 + expected_fy_thrust

        self.assertAlmostEqual(fx_total, expected_fx_total)
        self.assertAlmostEqual(fy_total, expected_fy_total)

    def test_simulation_step_after_thrust_duration(self):
        gravity = Gravity(MASS, ENVIRONMENT.gravity)
        drag = Drag(ENVIRONMENT.drag_coefficient, ENVIRONMENT.air_density, FRONTAL_AREA)
        thrust = InitialThrust(force=100.0, duration=2.0, direction=math.radians(45.0))
        state = State(t=3, x=0, y=0, vx=10, vy=0)  # Time is after the thrust duration
        fx_gravity, fy_gravity = gravity.calculate(state)
        fx_drag, fy_drag = drag.calculate(state)
        fx_thrust, fy_thrust = thrust.calculate(state)

        fx_total = fx_gravity + fx_drag + fx_thrust
        fy_total = fy_gravity + fy_drag + fy_thrust

        expected_fx_thrust = 0.0  # Thrust has ended
        expected_fy_thrust = 0.0
        expected_fx_total = 0.0 + (-0.5 * ENVIRONMENT.drag_coefficient * ENVIRONMENT.air_density * FRONTAL_AREA * 10**2) + expected_fx_thrust
        expected_fy_total = -MASS * ENVIRONMENT.gravity + 0.0 + expected_fy_thrust

        self.assertAlmostEqual(fx_total, expected_fx_total)
        self.assertAlmostEqual(fy_total, expected_fy_total)

    def test_simulation_run_until_stop_condition(self):
        gravity = Gravity(MASS, ENVIRONMENT.gravity)
        drag = Drag(ENVIRONMENT.drag_coefficient, ENVIRONMENT.air_density, FRONTAL_AREA)
        thrust = InitialThrust(force=100.0, duration=2.0, direction=math.radians(45.0))
        state = State(t=0, x=0, y=0, vx=10, vy=0)  # Time is after the thrust duration
        states = ForwardEulerSimulator(
                    force=CombinedForces([gravity, drag, thrust]),
                    dt=DELTA_T,
                    mass=MASS,
                    stop_condition=STOP_CONDITION_HIT_GROUND
                ).simulate(State(t=0, x=0, y=0, vx=10, vy=0))

        self.assertAlmostEqual(states[-1].x, 1477.1969881440098)  # The object should be approximately at the ground level
        self.assertTrue(states[-2].y >= 0)  # The object did not hit the ground yet
        self.assertTrue(states[-1].y <= 0)  # The object should have hit the ground