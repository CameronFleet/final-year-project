import builder, config
import math

class BoosterAgent:
    def __init__(self, world, W, H, np_random):
        self.body = builder.generate_booster(world, W, H, np_random)
        self.np_random = np_random

    def fireMainEngine(self, m_power, create_particle):
        tip, side, dispersion = self.calcMetrics()

        ox = tip[0] * (10 + 2 * dispersion[0]) + side[0] * dispersion[1]  # 4 is move a bit downwards, +-2 for randomness
        oy = -tip[1] * (3 + 2 * dispersion[0]) - side[1] * dispersion[1]
        impulse_pos = (self.body.position[0] + ox, self.body.position[1] + oy)

        if config.MAIN_ENGINE_POWER:
            p = create_particle(3.5, impulse_pos[0], impulse_pos[1], m_power)  # particles are just a decoration, 3.5 is here to make particle speed adequate
            p.ApplyLinearImpulse((ox * config.MAIN_ENGINE_POWER * m_power, oy * config.MAIN_ENGINE_POWER * m_power), impulse_pos,True)

        self.body.ApplyLinearImpulse((-ox * config.MAIN_ENGINE_POWER * m_power, -oy * config.MAIN_ENGINE_POWER * m_power),
                                        impulse_pos, True)

    def fireSideEngine(self, s_power, direction, create_particle):
        tip, side, dispersion = self.calcMetrics()

        s_power = 1.0
        ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * config.SIDE_ENGINE_AWAY )
        oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * config.SIDE_ENGINE_AWAY )
        impulse_pos = (self.body.position[0] + ox - tip[0] * 17 ,
                        self.body.position[1] + oy + tip[1] * config.SIDE_ENGINE_HEIGHT)
        if config.SIDE_ENGINE_POWER:
            p = create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse((ox * config.SIDE_ENGINE_POWER * s_power, oy * config.SIDE_ENGINE_POWER * s_power), impulse_pos,
                                    True)
        self.body.ApplyLinearImpulse((-ox * config.SIDE_ENGINE_POWER * s_power, -oy * config.SIDE_ENGINE_POWER * s_power),
                                        impulse_pos, True)


    def calcMetrics(self): 
        tip = (math.sin(self.body.angle), math.cos(self.body.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) for _ in range(2)]
        return tip, side, dispersion