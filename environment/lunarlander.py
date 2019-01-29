import sys, math

import builder  

import Box2D
from Box2D.b2 import circleShape

from detector import ContactDetector
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

import config 

class LunarLander(gym.Env, EzPickle):
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
        self.lander = None
        self.particles = []

        self.prev_reward = None

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)

        if self.continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        else:
            # Nop, fire left engine, main engine, right engine
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
        self.world.DestroyBody(self.lander)
        self.lander = None
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
        CHUNKS = 10
        self.terrian, self.pad = builder.generate_terrian(self.world, W, H)

        # GENERATE HELIPAD POLES
        self.helipad_x1 = W/2 - config.GOAL_W/2
        self.helipad_x2 = W/2 + config.GOAL_W/2
        self.helipad_y = H/4 + config.GOAL_H

        # GENERATE LANDER
        self.lander = builder.generate_booster(self.world, W, H, self.np_random)

        # GENERATE LEGS
        self.legs = builder.generate_landing_legs(self.world, W, H, self.lander)

        self.drawlist = [self.lander, self.terrian, self.pad] + self.legs

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
    def step(self, action):
        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        else:
            assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))

        # Engines
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) for _ in range(2)]

        # Main engine
        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (not self.continuous and action == 2):
           
            if self.continuous:
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0
            ox = tip[0] * (4 + 2 * dispersion[0]) + side[0] * dispersion[
                1]  # 4 is move a bit downwards, +-2 for randomness
            oy = -tip[1] * (4 + 2 * dispersion[0]) - side[1] * dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1],
                                      m_power)  # particles are just a decoration, 3.5 is here to make particle speed adequate
            p.ApplyLinearImpulse((ox * config.MAIN_ENGINE_POWER * m_power, oy * config.MAIN_ENGINE_POWER * m_power), impulse_pos,
                                 True)
            self.lander.ApplyLinearImpulse((-ox * config.MAIN_ENGINE_POWER * m_power, -oy * config.MAIN_ENGINE_POWER * m_power),
                                           impulse_pos, True)

        # Orientation engines
        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (not self.continuous and action in [1, 3]):
      
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                direction = action - 2
                s_power = 1.0
            ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * config.SIDE_ENGINE_AWAY )
            oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * config.SIDE_ENGINE_AWAY )
            impulse_pos = (self.lander.position[0] + ox - tip[0] * 17 ,
                           self.lander.position[1] + oy + tip[1] * config.SIDE_ENGINE_HEIGHT)
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse((ox * config.SIDE_ENGINE_POWER * s_power, oy * config.SIDE_ENGINE_POWER * s_power), impulse_pos,
                                 True)
            self.lander.ApplyLinearImpulse((-ox * config.SIDE_ENGINE_POWER * s_power, -oy * config.SIDE_ENGINE_POWER * s_power),
                                           impulse_pos, True)

        # Step a reasonable amount in Box2D
        self.world.Step(1.0 / config.FPS, 6 * 30, 2 * 30)

        # Update state
        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            (pos.x - config.VIEWPORT_W / 2) / (config.VIEWPORT_W/ 2),
            (pos.y - (self.helipad_y + config.LEG_DOWN )) / (config.VIEWPORT_H / 2),
            vel.x * (config.VIEWPORT_W / 2) / config.FPS,
            vel.y * (config.VIEWPORT_H / 2) / config.FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / config.FPS,
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

        reward -= m_power * 0.30  # less fuel spent is better, about -30 for heurisic landing
        reward -= s_power * 0.03

        # See if state is done
        done = False
        if self.game_over or abs(state[0]) >= 1.0:
            done = True
            reward = -100
        if not self.lander.awake:
            done = True
            reward = +100
        return np.array(state, dtype=np.float32), reward, done, {}

    def render(self, mode='human'):

        # (l,b), (l,t), (r,t), (r,b)
        from gym.envs.classic_control import rendering
        
        # Gym Viewer 
        if self.viewer is None:
            self.viewer = rendering.Viewer(config.VIEWPORT_W, config.VIEWPORT_H)
            self.viewer.set_bounds(0, config.WORLD_W, 0, config.WORLD_H)

            @self.viewer.window.event
            def on_key_press(symbol, modifiers):
                print('A key was pressed')

        # Degrade exhaust
        for obj in self.particles:
            obj.ttl -= 0.15
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
            flagy1 = self.helipad_y + 1 
            flagy2 = flagy1 - 31 
            self.viewer.draw_polygon([(x-1, flagy2), (x-1, flagy1), (x + 20, flagy1), (x + 20, flagy2)],
                                     color=(0.8, 0.8, 0))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class LunarLanderContinuous(LunarLander):
    continuous = True
