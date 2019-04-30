import pytest
from environment import BoosterLander
import environment.physics as physics
import environment.logic as logic
from Box2D import b2Vec2 as Vec2
import math

class TestEnvironment(object):

    @pytest.fixture
    def body(self):
        body = lambda: None
        body.position = [100,100]
        body.angle    = 0.5
        return body
        
    @pytest.fixture
    def make_body(self):

        def _make_body(vel, tilt):
            body = lambda: None
            body.linearVelocity = Vec2(vel[0],vel[1])
            body.worldCenter    = Vec2(500,500)
            body.diameter       = 5
            body.height         = 20
            body.angle          = tilt
            return body

        return _make_body

    @pytest.fixture
    def make_env(self):

        def _make_env(game_over = False, done=False, x=50, y=50, leg_contact=False, steps=1):
            env = BoosterLander()
            env.reset()
            env.steps = steps
            env.booster.body.position = Vec2(x,y)
            env.legs[0].ground_contact = leg_contact
            env.legs[1].ground_contact = leg_contact
            env.game_over = game_over
            env.done = done
            return env

        return _make_env

    @pytest.fixture
    def make_booster(self):

        def _make_booster(awake):
            body = lambda: None
            body.awake = awake

            booster = lambda: None
            booster.body = body
            
            return booster

        return _make_booster

    # drag_force
    def test_drag_force_smallest_area(self, make_body):
        body = make_body((0,-100), 0)
        air_density = 1
        drag_constant = 1
        drag, cog = physics.drag_force(body, air_density, drag_constant)

        expected_drag = (0,125000)
        expected_cog  = Vec2(500,500)

        assert expected_drag == drag and expected_cog == cog

    def test_drag_force_biggest_area(self, make_body):
        body = make_body((0,-100), math.pi/2)
        air_density = 1
        drag_constant = 1
        drag, cog = physics.drag_force(body, air_density, drag_constant)

        expected_drag = (0,500000)
        expected_cog  = Vec2(500,500)

        assert expected_drag == drag and expected_cog == cog

    # episode_complete
    def test_episode_complete_from_game_over(self, make_booster, make_env):
        env = make_env(game_over=True)
        done, landed, _, _  = logic.episode_complete(env)

        assert done == True and landed == False 

    def test_episode_complete_from_out_of_bounds(self, make_booster, make_env):
        env = make_env(x=-100, y=-100)
        done, landed, _, _  = logic.episode_complete(env)

        assert done == True and landed == False 

    def test_episode_complete_from_landing(self, make_booster, make_env):
        env = make_env(leg_contact=True)
        done, landed, _, _  = logic.episode_complete(env)

        assert done == True and landed == True 

    def test_episode_complete_from_manual_input(self, make_booster, make_env):
        env = make_env(done=True)
        done, landed, _, reward  = logic.episode_complete(env)

        assert done == True and landed == False and reward == 0

    def test_episode_complete_from_time_termination(self, make_booster, make_env):
        env = make_env(steps=50000)
        done, landed, _, _  = logic.episode_complete(env)

        assert done == True and landed == False

    # transform_engine_power
    def test_transform_engine_power_normal_operation(self):
        assert physics.transform_engine_power(300, 60) == 5

    def test_transform_engine_power_reduced_framerate(self):
        assert physics.transform_engine_power(300, 30) == 10

    def test_transform_engine_power_edge(self):
        assert physics.transform_engine_power(300, 1) == 300

    # engine_impulse
    def test_engine_impulse_0_degree(self):
        Fx, Fy = physics.engine_impulse(300, 0, 0, 0)
        assert Fx == 0 and Fy == 300

    def test_engine_impulse_45_degreee_with_dispersion(self):
        Fx, Fy = physics.engine_impulse(424.264068712, 0, 0.1, math.pi/4 - 0.1)
        expected_Fx = 300
        expected_Fy = 300
        assert round(Fx,5) == expected_Fx and round(Fy,5) == expected_Fy

    def test_engine_impulse_60_degree(self):
        Fx, Fy = physics.engine_impulse(424.264068712, math.pi/3, 0, 0)
        expected_Fx = -367.42346
        expected_Fy = 212.13203
        assert round(Fx,5) == expected_Fx and round(Fy,5) == expected_Fy

    def test_engine_impulse_60_degree_negative_force(self):
        Fx, Fy = physics.engine_impulse(-424.264068712, math.pi/3, 0, 0)
        expected_Fx = 367.42346
        expected_Fy = -212.13203
        assert round(Fx,5) == expected_Fx and round(Fy,5) == expected_Fy

    def test_engine_impulse_negative_60_degree(self):
        Fx, Fy = physics.engine_impulse(424.264068712, -math.pi/3, 0, 0)
        expected_Fx = 367.42346
        expected_Fy = 212.13203
        assert round(Fx,5) == expected_Fx and round(Fy,5) == expected_Fy

    def test_engine_impulse_90_degree_orientation(self):
        Fx, Fy = physics.engine_impulse(500, 0, 0, math.pi/2)
        expected_Fx = 500
        expected_Fy = 0
        assert round(Fx,5) == expected_Fx and round(Fy,5) == expected_Fy

    def test_engine_impulse_90_degree_orientation_with_dispersion(self):
        Fx, Fy = physics.engine_impulse(500, 0, 0.1, math.pi/2)
        expected_Fx = 497.50208
        expected_Fy = -49.91671
        assert round(Fx,5) == expected_Fx and round(Fy,5) == expected_Fy
    
    def test_engine_impulse_edge(self):
        Fx, Fy = physics.engine_impulse(0, math.pi, math.pi, math.pi)
        expected_Fx = 0
        expected_Fy = 0
        assert round(Fx,5) == expected_Fx and round(Fy,5) == expected_Fy

    # side_engine_impulse_position 
    def test_side_engine_impulse_position_left_thruster(self, body):
        impulse_position = physics.side_engine_impulse_position(body, -1, 40, 3)
        expected_impulse = (79.50660, 134.38416)
        assert round(impulse_position[0],5) == expected_impulse[0] and round(impulse_position[1],5) == expected_impulse[1]
    
    def test_side_engine_impulse_position_right_thruster(self, body):
        impulse_position = physics.side_engine_impulse_position(body, 1, 40, 3)
        expected_impulse = (82.13935, 135.82244)
        assert round(impulse_position[0],5) == expected_impulse[0] and round(impulse_position[1],5) == expected_impulse[1]
    
    def test_side_engine_impulse_position_edge(self, body):
        impulse_position = physics.side_engine_impulse_position(body, 1, 0, 0)
        expected_impulse = (100.0, 100.0)
        assert round(impulse_position[0],5) == expected_impulse[0] and round(impulse_position[1],5) == expected_impulse[1]
