import environment.builder as builder
import environment.config as config
import math

from environment.physics import transform_engine_power, engine_impulse, side_engine_impulse_position

class Booster:
    def __init__(self, world, W, H, sensor, initial_random, np_random):
        self.body = builder.generate_booster(world, W, H, np_random)
        self.body.diameter  = config.LANDER_DIAMETER
        self.body.height    = config.LANDER_HEIGHT
        self.mfr = config.MASS_FLOW_RATE
        self.np_random = np_random
        self.MAIN_ENGINE_POWER = transform_engine_power(config.MAIN_ENGINE_POWER, config.FPS)
        self.SIDE_ENGINE_POWER = transform_engine_power(config.SIDE_ENGINE_POWER, config.FPS)
        self.sensor = sensor

        self.body.ApplyForceToCenter((
        np_random.uniform(-initial_random, initial_random)*1000,
        np_random.uniform(-initial_random, initial_random)*1000
        ), True)

    def fireMainEngine(self, m_power, alpha, create_particle, record_metrics):

        dispersion = self.np_random.uniform(-0.1, +0.1)
        impulse_pos = (self.body.position[0], self.body.position[1])

        if self.MAIN_ENGINE_POWER:
            p = create_particle(3.5, impulse_pos[0], impulse_pos[1], m_power)  # particles are just a decoration, 3.5 is here to make particle speed adequate
            p.coldGas = False  

            Fx, Fy = engine_impulse(self.MAIN_ENGINE_POWER * m_power, self.body.angle, alpha, dispersion)
            record_metrics({"Ft":math.sqrt(Fx*Fx + Fy*Fy), "alpha":alpha}, "actions")

            p.ApplyLinearImpulse((-Fx, -Fy), impulse_pos,True)
            self.body.ApplyLinearImpulse((Fx, Fy),
                                            impulse_pos, True)
            return Fx, Fy, impulse_pos

    def fireSideEngine(self, s_power, direction, create_particle, record_metrics):

        if self.sensor.sense() == 2 and direction > 0:
            record_metrics({"Fs":0},"actions")
            return None
        elif self.sensor.sense() == 1 and direction < 0:
            record_metrics({"Fs":0},"actions")
            return None
        
        dispersion = self.np_random.uniform(-0.1, +0.1)
        impulse_pos = side_engine_impulse_position(self.body, direction, config.SIDE_ENGINE_HEIGHT, config.SIDE_ENGINE_AWAY)

        if self.SIDE_ENGINE_POWER:
            p = create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.coldGas = True

            Fx, Fy = engine_impulse(direction*self.SIDE_ENGINE_POWER * s_power, 
                                    self.body.angle, 
                                    dispersion=dispersion, 
                                    orientation=math.pi/2)

            record_metrics({"Fs":math.sqrt(Fx*Fx + Fy*Fy)},"actions")
            
            p.ApplyLinearImpulse((Fx, Fy), impulse_pos, True)
            self.body.ApplyLinearImpulse((-Fx, -Fy), impulse_pos, True)





