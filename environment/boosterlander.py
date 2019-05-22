import sys, math

import environment.builder as builder
from environment.booster import Booster
from environment.detector import ContactDetector
from environment.sensor import Sensor

import Box2D
from Box2D.b2 import circleShape
from Box2D import b2Vec2 as Vec2

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding, EzPickle

import environment.config as config

import time

from environment.physics import drag_force, impulse 
from environment.logic import episode_complete 
from util import discretization_actions




class BoosterLander(gym.Env, EzPickle):
    """
    Observation Space
    ( 
    x,                        (0 ---> WORLD_W)
    y,                        (0 ---> WORLD_H)
    vx,                       (-110 ---> 110)
    vy,                       (-110 ---> 110)
    ф,                        (-4п ---> 4п)
    vф,                       (-4п ---> 4п)
    leg[0] contact,           ({0,1})
    leg[1] contact            ({0,1})

    Action Space
    action    Ft, alpha, Fs
    0         (0,0,0)
    1         (1.0, 0, 0)
    2         (0.5, 0, 0)
        ...
    28        (1.0, -0.05, 0.5)
    29        (0.5, 0.05, 0.5)
    30        (0.5, -0.05, 0.5)
    31        (0,0,1.0)
    32        (0,0,-1.0)
    33        (0,0,0.5)
    34        (0,0,-0.5)
    )
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': config.FPS
    }

    initial_random = 1000.0
    continuous = False

    good_accelerometer = True
    good_gps = True
    good_rate_sensor = True
    good_side_thrusters = True
    sensors_in_observation = False


    def __init__(self, seed=None, time_terminated=True, moving_goal =False, termination_time=1000):
        EzPickle.__init__(self)

        self.viewer = None
        
        self.world = Box2D.b2World()
        self.T = config.FPS
        self.terrian = None
        self.booster = None
        self.particles = []

        self.prev_reward = None
        self.user_action = (0,0,0)

        self.steps = 0
        self.done = False
        self.time_terminated = time_terminated

        self.tracked_metrics = {}

        self.np_random, self.seed = seeding.np_random(seed)

    
        low = [0, 0, -120, -120, -4*math.pi, -4*math.pi, 0, 0]
        high = [config.WORLD_W, config.WORLD_H, 120, 120, 4*math.pi, 4*math.pi, 1, 1]

        if self.sensors_in_observation:
            low += [0,0,0,0]
            high += [2,2,2,2]

        self.observation_space = spaces.Box(np.array(low), np.array(high), dtype=np.float32)

        # self.actions = discretization_actions(1,1,1)
        self.actions = [(0,0,0), (1.0, 0, 0), (1.0, -0.1, 0), (1.0, 0.1, 0), (0,0,-1.0), (0,0,1.0)]
        # self.actions = [(0.2,0,0), (1.0, 0, 0), (1.0, -0.1, 0), (1.0, 0.1, 0), (0.2, -0.1, 0), (0.2, 0.1, 0), (0.2,0,-1.0), (0.2,0,1.0), (1,0,-1.0), (1,0,1.0)]
        self.action_space = spaces.Discrete(len(self.actions))

        self.moving_goal = moving_goal
        self.termination_time = termination_time

        self.reset()

    def _destroy(self):
        if not self.terrian: return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.terrian)
        self.world.DestroyBody(self.pad)
        self.terrian = None
        self.world.DestroyBody(self.booster.body)
        self.booster = None
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

        #  Decide the goal
        scale = self.np_random.uniform(config.GOAL_MIN_X, config.GOAL_MAX_X) if self.moving_goal else config.GOAL_X_SCALED
        self.GOAL = [W*scale, config.SEA_LEVEL + config.GOAL_H]

        # GENERATE HELIPAD POLES
        self.helipad_x1 = self.GOAL[0] - config.GOAL_W/2
        self.helipad_x2 = self.GOAL[0] + config.GOAL_W/2
        self.helipad_y = config.SEA_LEVEL + config.GOAL_H

        # GENERATE TERRIAN
        self.terrian, self.pad = builder.generate_terrian(self.world, W, H, self.helipad_x1, self.helipad_x2, self.helipad_y)

        # GENERATE booster
        # self.failed_side_thrusters = not self.good_side_thrusters and self.np_random.uniform(0,1) < config.SIDE_BOOSTER_FAILURE_CHANCE
        self.side_thrusters_sensor = Sensor(None, 
                                            self.good_side_thrusters,
                                            config.SIDE_BOOSTER_FAILURE_CHANCE,
                                            self.np_random)
        self.booster = Booster(self.world, W, H, self.side_thrusters_sensor, self.initial_random, self.np_random)

        # GENERATE LEGS
        self.legs = builder.generate_landing_legs(self.world, W, H, self.booster.body)

        self.drawlist = [self.booster.body, self.terrian, self.pad] + self.legs
        self.steps = 0

        # Attach sensors
        self.accelerometer = Sensor(self.booster.body.linearVelocity, 
                                    self.good_accelerometer, 
                                    config.ACCELEROMETER_FAILURE_CHANCE, 
                                    self.np_random)
        self.gps           = Sensor(self.booster.body.position,
                                    self.good_gps,
                                    config.GPS_FAILURE_CHANCE,
                                    self.np_random)
        self.roll_rate    = Sensor(lambda : (self.booster.body.angle, self.booster.body.angularVelocity),
                                    self.good_rate_sensor,
                                    config.ROLL_FAILURE_CHANCE,
                                    self.np_random,
                                    functor=True)
        print(self.roll_rate.failure_code())


        return self.step(None)[0]

    def _create_particle(self, mass, x, y, ttl):
        p = builder.generate_particle(self.world, x, y, mass)
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all):
        while self.particles and (all or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def _apply_drag(self, body):
        drag, cog = drag_force(body, config.SEA_LEVEL_DENSITY, 0.75)
        body.ApplyForce((drag[0], drag[1]), (cog.x, cog.y), False)
        self._record_metrics({"dragForce_x": drag[0], "dragForce_y": drag[1]})

    def _record_metrics(self, metrics, group=None):
        if group is None:
            if "misc" in self.tracked_metrics: 
                self.tracked_metrics["misc"] = {**self.tracked_metrics["misc"], **metrics}
            else:
                self.tracked_metrics["misc"] = metrics
        else: 
            if group in self.tracked_metrics: 
                self.tracked_metrics[group] = {**self.tracked_metrics[group], **metrics}
            else:
                self.tracked_metrics[group] = metrics

    def step(self, action):
        self.steps += 1
        self.tracked_metrics = {}

        if self.user_action != (0,0,0):
            action = self.user_action

        if action is not None:
            if self.continuous or self.user_action != (0,0,0):
                Ft, alpha, Fs = action
            else: 
                Ft, alpha, Fs = self.actions[action]
        
            if Ft:
                self.booster.fireMainEngine(Ft, alpha, self._create_particle, self._record_metrics)
            else:
                Ft = 0 
                self._record_metrics({"Ft":Ft, "alpha":alpha}, "actions")

            if Fs:
                self.booster.fireSideEngine(abs(Fs), Fs/abs(Fs), self._create_particle, self._record_metrics)
            else:
                Fs = 0
                self._record_metrics({"Fs":Fs}, "actions")
        else:
            Ft, alpha, Fs = 0, 0, 0

        self._apply_drag(self.booster.body)

        self.world.Step(1.0 / self.T, 6, 2)

        # Update state
        vel = Vec2(self.accelerometer.sense())
        pos = Vec2(self.gps.sense())
        roll = self.roll_rate.sense()
        state = [
                pos.x - self.GOAL[0],
                pos.y,
                vel.x,
                vel.y,
                roll[0],
                roll[1],
                1.0 if self.legs[0].ground_contact else 0.0,
                1.0 if self.legs[1].ground_contact else 0.0
            ]

        if self.sensors_in_observation:
            state += [
                self.accelerometer.failure_code(),
                self.gps.failure_code(),
                self.roll_rate.failure_code(),
                self.side_thrusters_sensor.failure_code()
            ]
            
     
        self._record_metrics({"x":state[0],
                        "y":state[1],
                        "vx":state[2],
                        "vy":state[3],
                        "theta":state[4],
                        "vtheta":state[5],
                        "leg_left":state[6],
                        "leg_right":state[7]}, "observation")

        if self.sensors_in_observation:
            assert len(state) == 12
        else:
            assert len(state) == 8

        """
        REWARD SCHEMES

        Shaped reward

        """
        reward = 0

        vel = self.booster.body.linearVelocity
        pos = self.booster.body.position
        angle = self.booster.body.angle

        x_diff = pos.x - self.GOAL[0] 
        y_diff = pos.y - self.GOAL[1]

        shaping = \
            -0.5*np.sqrt(x_diff*x_diff + y_diff*y_diff) \
            -np.sqrt(vel.x*vel.x + vel.y*vel.y) \
            -100*abs(angle) 

        if self.prev_shaping is not None: 
            reward = shaping - self.prev_shaping

        self.prev_shaping = shaping

        if action is not None:
            reward -= 0.3*Ft #was 0.1

        # See if state is done
        done, landed, impulse, completion_reward = episode_complete(self)    
        if completion_reward is not None:
            reward += completion_reward
            
        # Performance metrics used in evaluation
        performance_metrics = {}
        performance_metrics['Ft'] = Ft
        performance_metrics['Fs'] = Fs
        if done:
            performance_metrics['landed'] = landed
            performance_metrics['impulse'] = impulse

        return np.array(state, dtype=np.float32), reward, done, performance_metrics

    def render(self, metrics=True, mode='human'):
        import util.rendering as rendering
        import pyglet
        from pyglet.window import key

        self._record_metrics({  "accelerometer failure": self.accelerometer.failure_code(),
                                "gps failure":self.gps.failure_code(),
                                "roll rate failure":self.roll_rate.failure_code(),
                                "side thrusters failure":self.side_thrusters_sensor.failure_code()}, "observation" if self.sensors_in_observation else "sensors")
        # Create Viewer 
        if self.viewer is None:
            self.viewer = rendering.Viewer(config.VIEWPORT_W, config.VIEWPORT_H)
            self.viewer.set_bounds(0, config.WORLD_W, 0, config.WORLD_H)

            @self.viewer.window.event
            def on_key_press(symbol, modifiers):
                if symbol == key.UP:
                    self.user_action = (1, self.user_action[1], self.user_action[2])
                if symbol == key.LEFT: 
                    self.user_action = (self.user_action[0], -0.1, self.user_action[2])
                if symbol == key.RIGHT:
                    self.user_action = (self.user_action[0], 0.1, self.user_action[2])
                if symbol == key.Q:
                    self.viewer.close()
                    self.done = True
                if symbol == key.R:
                    self.reset()

            @self.viewer.window.event
            def on_key_release(symbol, modifiers):
                if symbol == key.UP:
                    self.user_action = (0, self.user_action[1], self.user_action[2])
                if symbol == key.LEFT: 
                    self.user_action = (self.user_action[0], 0, self.user_action[2])
                if symbol == key.RIGHT:
                    self.user_action = (self.user_action[0], 0, self.user_action[2])

        # Draw metrics
        self.viewer.draw_fps()

        if metrics:
            for group in self.tracked_metrics:
                self.viewer.draw_heading(group)
                for metric in self.tracked_metrics[group]:
                    self.viewer.draw_metric(metric, self.tracked_metrics[group][metric])

        # Degrade exhaust
        for obj in self.particles:
            obj.ttl -= 0.05
            obj.color1 = (max(0.2, 0.2 + obj.ttl), max(0.2, 0.5 * obj.ttl), max(0.2, 0.5 * obj.ttl))
            obj.color2 = (max(0.2, 0.2 + obj.ttl), max(0.2, 0.5 * obj.ttl), max(0.2, 0.5 * obj.ttl))
            if obj.coldGas:
                obj.color1 = (max(0.2, 0.3 + obj.ttl), max(0.2, 0.3 + obj.ttl), max(0.2, 0.7 + obj.ttl))
                obj.color2 = (max(0.2, 0.3 + obj.ttl), max(0.2, 0.3 + obj.ttl), max(0.2, 0.7 + obj.ttl))
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
                                    

        # Position Indicator               
        # x,y = self.booster.body.position
        # s = 1
        # self.viewer.draw_polygon([(x-s,y-s),(x-s,y+s), (x+s, y+s), (x+s, y-s)], color=(0, 0, 0))        

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

class BoosterLanderContinuous(BoosterLander):
    continuous = True