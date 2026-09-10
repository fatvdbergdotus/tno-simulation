from solution import *
import unittest

class TestProjectile(unittest.TestCase):
    def test_projectile_properties(self):
        PROJECTILE = Projectile()
        self.assertEqual(PROJECTILE.radius, 0.03)
        self.assertEqual(PROJECTILE.density, 7800.0)
        self.assertEqual(PROJECTILE.initial_velocity, 200.0)
        self.assertEqual(PROJECTILE.launch_angle_rad, math.radians(45.0))

class TestEnvironment(unittest.TestCase):
    def test_enviroment_properties(self):
        ENVIRONMENT = Environment()
        self.assertEqual( ENVIRONMENT.gravity, 9.81)
        self.assertEqual( ENVIRONMENT.air_density, 1.225)
        self.assertEqual( ENVIRONMENT.drag_coefficient, 0.47)