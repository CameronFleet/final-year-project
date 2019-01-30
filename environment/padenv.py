import sys, math

import builder  
from boosteragent import BoosterAgent

import Box2D
from Box2D.b2 import circleShape

from detector import ContactDetector
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

import config 
import time

class PadEnv(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': config.FPS
    }

    continuous = False

    def __init__(self):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None
        
        self.world = Box2D.b2World()
        self.terrian = None
        self.agent = None
        self.particles = []

        self.prev_reward = None

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)

        self.action_space = spaces.Discrete(4)

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.terrian: return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.terrian)
        self.terrian = None
        self.world.DestroyBody(self.agent.body)
        self.agent = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])

    def reset(self):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        W = config.WORLD_W
        H = config.WORLD_H

        # GENERATE TERRIAN
        self.terrian, self.pad = builder.generate_terrian(self.world, W, H)

        # GENERATE HELIPAD POLES
        self.helipad_x1 = W/2 - config.GOAL_W/2
        self.helipad_x2 = W/2 + config.GOAL_W/2
        self.helipad_y = H/4 + config.GOAL_H

        # GENERATE AGENT
        self.agent = BoosterAgent(self.world, W, H, self.np_random)

        # GENERATE LEGS
        self.legs = builder.generate_landing_legs(self.world, W, H, self.agent.body)

        self.drawlist = [self.agent.body, self.terrian, self.pad] + self.legs

        return self.step(np.array([0, 0]) if self.continuous else 0)[0]

    # Exhaust create
    def _create_particle(self, mass, x, y, ttl):
        p = builder.generate_particle(self.world, x, y, mass)
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    # Exhaust delete
    def _clean_particles(self, all):
        while self.particles and (all or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    # One step in the envrionment
    # Equal to 1/FPS time step
    def step(self, action):

        # Main engine
        if action == 2:
            self.agent.fireMainEngine(1.0, self._create_particle)

        # Orientation engines
        if action == 1 or action == 3:
            self.agent.fireSideEngine(1.0, action - 2, self._create_particle)

        # Step a reasonable amount in Box2D
        self.world.Step(1.0 / config.FPS, 6, 2)

        # Update state
        pos = self.agent.body.position
        vel = self.agent.body.linearVelocity
        state = [
            (pos.x - config.VIEWPORT_W / 2) / (config.VIEWPORT_W/ 2),
            (pos.y - (self.helipad_y + config.LEG_DOWN )) / (config.VIEWPORT_H / 2),
            vel.x * (config.VIEWPORT_W / 2) / config.FPS,
            vel.y * (config.VIEWPORT_H / 2) / config.FPS,
            self.agent.body.angle,
            20.0 * self.agent.body.angularVelocity / config.FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0
        ]
        assert len(state) == 8

        # Calculate reward
        reward = 0
        shaping = \
            - 100 * np.sqrt(state[0] * state[0] + state[1] * state[1]) \
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3]) \
            - 100 * abs(state[4]) + 10 * state[6] + 10 * state[7]  # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # reward -= m_power * 0.30  # less fuel spent is better, about -30 for heurisic landing
        # reward -= s_power * 0.03

        # See if state is done
        done = False
        if self.game_over or abs(state[0]) >= 1.0:
            done = True
            reward = -100
        if not self.agent.body.awake:
            done = True
            reward = +100
        return np.array(state, dtype=np.float32), reward, done, {}

    def render(self, mode='human'):
    
        # (l,b), (l,t), (r,t), (r,b)
        import rendering
        import pyglet

        # Create Viewer 
        if self.viewer is None:
            self.viewer = rendering.Viewer(config.VIEWPORT_W, config.VIEWPORT_H)
            self.viewer.set_bounds(0, config.WORLD_W, 0, config.WORLD_H)

            @self.viewer.window.event
            def on_key_press(symbol, modifiers):
                print('A key was pressed')

        self.viewer.draw_fps()
        self.viewer.draw_metric("V_i", self.agent.body.linearVelocity[0])
        self.viewer.draw_metric("V_j", self.agent.body.linearVelocity[1])

        # Degrade exhaust
        for obj in self.particles:
            obj.ttl -= 0.05
            obj.color1 = (max(0.2, 0.2 + obj.ttl), max(0.2, 0.5 * obj.ttl), max(0.2, 0.5 * obj.ttl))
            obj.color2 = (max(0.2, 0.2 + obj.ttl), max(0.2, 0.5 * obj.ttl), max(0.2, 0.5 * obj.ttl))
        self._clean_particles(False)

        # Render Sky
        self.viewer.draw_polygon([(0,0),(0,config.WORLD_H), (config.WORLD_W, config.WORLD_H), (config.WORLD_W, 0)], color=(0.2, 0.8, 1))

        # Render fixtures? 
        for obj in self.particles + self.drawlist:

            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        # Render helipad falgs
        for x in [self.helipad_x1, self.helipad_x2]:
            flagy1 = self.helipad_y +3
            flagy2 = self.helipad_y - config.GOAL_H - 1
            self.viewer.draw_polygon([(x-1, flagy2), (x-1, flagy1), (x + config.GOAL_W/10, flagy1), (x + config.GOAL_W/10, flagy2)],
                                     color=(0.8, 0.8, 0))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

