import pytest
import math
import project.environment.boosteragent as agent

class TestAgent(object):

    @pytest.fixture
    def body(self):
        body = lambda: None
        body.position = [100,100]
        body.angle    = 0.5
        return body

    # transform_engine_power
    def test_transform_engine_power_normal_operation(self):
        assert agent.transform_engine_power(300, 60) == 5

    def test_transform_engine_power_reduced_framerate(self):
        assert agent.transform_engine_power(300, 30) == 10

    def test_transform_engine_power_edge(self):
        assert agent.transform_engine_power(300, 1) == 300

    # engine_impulse
    def test_engine_impulse_0_degree(self):
        Fx, Fy = agent.engine_impulse(300, 0, 0, 0)
        assert Fx == 0 and Fy == 300

    def test_engine_impulse_45_degreee_with_dispersion(self):
        Fx, Fy = agent.engine_impulse(424.264068712, 0, 0.1, math.pi/4 - 0.1)
        expected_Fx = 300
        expected_Fy = 300
        assert round(Fx,5) == expected_Fx and round(Fy,5) == expected_Fy

    def test_engine_impulse_60_degree(self):
        Fx, Fy = agent.engine_impulse(424.264068712, math.pi/3, 0, 0)
        expected_Fx = -367.42346
        expected_Fy = 212.13203
        assert round(Fx,5) == expected_Fx and round(Fy,5) == expected_Fy

    def test_engine_impulse_60_degree_negative_force(self):
        Fx, Fy = agent.engine_impulse(-424.264068712, math.pi/3, 0, 0)
        expected_Fx = 367.42346
        expected_Fy = -212.13203
        assert round(Fx,5) == expected_Fx and round(Fy,5) == expected_Fy

    def test_engine_impulse_negative_60_degree(self):
        Fx, Fy = agent.engine_impulse(424.264068712, -math.pi/3, 0, 0)
        expected_Fx = 367.42346
        expected_Fy = 212.13203
        assert round(Fx,5) == expected_Fx and round(Fy,5) == expected_Fy

    def test_engine_impulse_90_degree_orientation(self):
        Fx, Fy = agent.engine_impulse(500, 0, 0, math.pi/2)
        expected_Fx = 500
        expected_Fy = 0
        assert round(Fx,5) == expected_Fx and round(Fy,5) == expected_Fy

    def test_engine_impulse_90_degree_orientation_with_dispersion(self):
        Fx, Fy = agent.engine_impulse(500, 0, 0.1, math.pi/2)
        expected_Fx = 497.50208
        expected_Fy = -49.91671
        assert round(Fx,5) == expected_Fx and round(Fy,5) == expected_Fy
    
    def test_engine_impulse_edge(self):
        Fx, Fy = agent.engine_impulse(0, math.pi, math.pi, math.pi)
        expected_Fx = 0
        expected_Fy = 0
        assert round(Fx,5) == expected_Fx and round(Fy,5) == expected_Fy

    # side_engine_impulse_position 
    def test_side_engine_impulse_position_left_thruster(self, body):
        impulse_position = agent.side_engine_impulse_position(body, -1, 40, 3)
        expected_impulse = (79.50660, 134.38416)
        assert round(impulse_position[0],5) == expected_impulse[0] and round(impulse_position[1],5) == expected_impulse[1]
    
    def test_side_engine_impulse_position_right_thruster(self, body):
        impulse_position = agent.side_engine_impulse_position(body, 1, 40, 3)
        expected_impulse = (82.13935, 135.82244)
        assert round(impulse_position[0],5) == expected_impulse[0] and round(impulse_position[1],5) == expected_impulse[1]
    
    def test_side_engine_impulse_position_edge(self, body):
        impulse_position = agent.side_engine_impulse_position(body, 1, 0, 0)
        expected_impulse = (100.0, 100.0)
        assert round(impulse_position[0],5) == expected_impulse[0] and round(impulse_position[1],5) == expected_impulse[1]
