import environment.builder as builder
import environment.config as config
import math

def transform_engine_power(power, fps):
    return power * 1/fps

def engine_impulse(power, body_tilt, gimbal = 0, dispersion = 0, orientation = 0):
    angle = -body_tilt +gimbal + dispersion
    Ft = power
    Fx = Ft*math.sin(angle + orientation)
    Fy = Ft*math.cos(angle + orientation)
    return Fx, Fy

def side_engine_impulse_position(body, direction, engine_height, engine_away):
    return (body.position[0] 
                - engine_height*math.sin(body.angle)
                + direction*(engine_away/2)*math.cos(body.angle), 
            body.position[1] 
                + engine_height*math.cos(body.angle)
                + direction*(engine_away/2)*math.sin(body.angle))

class Booster:
    def __init__(self, world, W, H, np_random):
        self.body = builder.generate_booster(world, W, H, np_random)
        self.body.diameter  = config.LANDER_DIAMETER
        self.body.height    = config.LANDER_HEIGHT
        self.np_random = np_random
        self.MAIN_ENGINE_POWER = transform_engine_power(config.MAIN_ENGINE_POWER, config.FPS)
        self.SIDE_ENGINE_POWER = transform_engine_power(config.SIDE_ENGINE_POWER, config.FPS)
        self.body.ApplyForceToCenter((
        np_random.uniform(-config.INITIAL_RANDOM, config.INITIAL_RANDOM)*1000,
        np_random.uniform(-config.INITIAL_RANDOM, config.INITIAL_RANDOM)*1000
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





