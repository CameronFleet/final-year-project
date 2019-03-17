import pytest
import project.environment.environment as environment
from Box2D import b2Vec2 as Vec2
import math

class TestEnvironment(object):

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

        def _make_env(game_over, done):
            env = lambda: None
            env.game_over = game_over
            env.done      = done
            return env

        return _make_env

    @pytest.fixture
    def make_agent(self):

        def _make_agent(awake):
            body = lambda: None
            body.awake = awake

            agent = lambda: None
            agent.body = body
            
            return agent

        return _make_agent

    # drag_force
    def test_drag_force_smallest_area(self, make_body):
        body = make_body((0,-100), 0)
        air_density = 1
        drag_constant = 1
        drag, cog = environment.drag_force(body, air_density, drag_constant)

        expected_drag = (0,125000)
        expected_cog  = Vec2(500,500)

        assert expected_drag == drag and expected_cog == cog

    def test_drag_force_biggest_area(self, make_body):
        body = make_body((0,-100), math.pi/2)
        air_density = 1
        drag_constant = 1
        drag, cog = environment.drag_force(body, air_density, drag_constant)

        expected_drag = (0,500000)
        expected_cog  = Vec2(500,500)

        assert expected_drag == drag and expected_cog == cog

    # episode_complete
    def test_episode_complete_from_game_over_success(self, make_env, make_agent):
        env   = make_env(True, False)
        agent = make_agent(False)

        done, reward = environment.episode_complete(agent, env)

        assert done == True and reward == 100

    def test_episode_complete_from_game_over_failure(self, make_env, make_agent):
        env   = make_env(True, False)
        agent = make_agent(True)

        done, reward = environment.episode_complete(agent, env)

        assert done == True and reward == -100

    def test_episode_complete_from_game_over_forced(self, make_env, make_agent):
        env   = make_env(False, True)
        agent = make_agent(False)

        done, reward = environment.episode_complete(agent, env)

        assert done == True and reward == 0

    def test_episode_complete_episode_continues(self, make_env, make_agent):
        env   = make_env(False, False)
        agent = make_agent(True)

        done, reward = environment.episode_complete(agent, env)

        assert done == False and reward == None

        
    def test_discretization_actions(self):
        #TODO: Write this test! for discretization of 1,1,1 and 2,1,2
        assert(1 ==2)