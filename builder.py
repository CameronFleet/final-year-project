import numpy as np
from Box2D.b2 import (edgeShape, fixtureDef, polygonShape, revoluteJointDef, circleShape)
import config 

initial_y = config.VIEWPORT_H / config.SCALE


def generate_terrian(world, W, H, CHUNKS, np_random):

    height = np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
    chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
    helipad_y = H / 4
    height[CHUNKS // 2 - 2] = helipad_y
    height[CHUNKS // 2 - 1] = helipad_y
    height[CHUNKS // 2 + 0] = helipad_y
    height[CHUNKS // 2 + 1] = helipad_y
    height[CHUNKS // 2 + 2] = helipad_y
    smooth_y = [0.33 * (height[i - 1] + height[i + 0] + height[i + 1]) for i in range(CHUNKS)]

    terrian = world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
    terrian.color1 = (0.0, 0.0, 0.0)
    terrian.color2 = (0.0, 0.0, 0.0)

    sky_polys = []
    for i in range(CHUNKS - 1):
        p1 = (chunk_x[i], smooth_y[i])
        p2 = (chunk_x[i + 1], smooth_y[i + 1])
        terrian.CreateEdgeFixture(
            vertices=[p1, p2],
            density=0,
            friction=0.1)
        sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

    return (sky_polys, terrian)

def generate_booster(world, np_random): 
    booster = world.CreateDynamicBody(
        position=(config.VIEWPORT_W / config.SCALE / 2, initial_y),
        angle=0.0,
        fixtures=fixtureDef(
            shape=polygonShape(vertices=[(x / config.SCALE, y / config.SCALE) for x, y in config.LANDER_POLY]),
            density=5.0,
            friction=0.1,
            categoryBits=0x0010,
            maskBits=0x001,  # collide only with ground
            restitution=0.0)  # 0.99 bouncy
    )
    booster.color1 = (0.5, 0.4, 0.9)
    booster.color2 = (0.3, 0.3, 0.5)
    booster.ApplyForceToCenter((
        np_random.uniform(-config.INITIAL_RANDOM, config.INITIAL_RANDOM),
        np_random.uniform(-config.INITIAL_RANDOM, config.INITIAL_RANDOM)
    ), True)

    return booster


def generate_landing_legs(world, booster):
    legs = []
    for i in [-1, +1]:
        leg = world.CreateDynamicBody(
            position=(config.VIEWPORT_W / config.SCALE / 2 - i * config.LEG_AWAY / config.SCALE, initial_y),
            angle=(i * 0.05),
            fixtures=fixtureDef(
                shape=polygonShape(box=(config.LEG_W / config.SCALE, config.LEG_H / config.SCALE)),
                density=1.0,
                restitution=0.0,
                categoryBits=0x0020,
                maskBits=0x001)
        )
        leg.ground_contact = False
        leg.color1 = (0.5, 0.4, 0.9)
        leg.color2 = (0.3, 0.3, 0.5)
        rjd = revoluteJointDef(
            bodyA=booster,
            bodyB=leg,
            localAnchorA=(0, 0),
            localAnchorB=(i * config.LEG_AWAY / config.SCALE, config.LEG_DOWN / config.SCALE),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=config.LEG_SPRING_TORQUE,
            motorSpeed=+0.3 * i  # low enough not to jump back into the sky
        )
        if i == -1:
            rjd.lowerAngle = +0.9 - 0.5  # Yes, the most esoteric numbers here, angles legs have freedom to travel within
            rjd.upperAngle = +0.9
        else:
            rjd.lowerAngle = -0.9
            rjd.upperAngle = -0.9 + 0.5
        leg.joint = world.CreateJoint(rjd)
        legs.append(leg)

    return legs


def generate_particle(world, x, y, mass):
    p = world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=2 / config.SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3)
        )

    return p