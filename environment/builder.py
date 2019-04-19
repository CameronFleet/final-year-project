import numpy as np
from Box2D.b2 import (circleShape, edgeShape, fixtureDef, polygonShape,
                      revoluteJointDef)

import environment.config as config

sea_level = config.SEA_LEVEL

def generate_terrian(world, W, H, g_x1, g_x2, g_y):

    pad = world.CreateStaticBody(
    position=(0,0),
    shapes=polygonShape(vertices=[(g_x1, sea_level),(g_x1, g_y), (g_x2, g_y), [g_x2, sea_level]])
    )

    pad.color1 = (0.4,0.4,0.4)
    pad.color2 = (0.4,0.4,0.4)

    terrian = world.CreateStaticBody(
    position=(0,0),
    shapes=polygonShape(vertices=[(0, 0),(0, sea_level), (W, sea_level), [W, 0]]),
    )


    terrian.color1 = (0.2,0.6,1)
    terrian.color2 = (0.2,0.6,1)

    return terrian, pad

def generate_booster(world, W, H, np_random): 
    booster = world.CreateDynamicBody(
        position=(W/2, H-60),
        angle=0,
        linearVelocity=config.START_VELOCITY,
        fixtures=fixtureDef(
            shape=polygonShape(vertices=[(x , y) for x, y in config.LANDER_POLY]),
            density=config.LANDER_DENSITY,
            friction=0.1,
            categoryBits=0x0010,
            maskBits=0x001,  # collide only with ground
            restitution=0.0)  # 0.99 bouncy
    )

    booster.color1 = (1, 1, 1)
    booster.color2 = (0.4, 0.4, 0.4)

    return booster

def generate_landing_legs(world, W, H, booster):
    legs = []
    for i in [-1, +1]:
        leg = world.CreateDynamicBody(
            position=(W/2- i * config.LEG_AWAY, H-60),
            angle=(i * 0.05),
            fixtures=fixtureDef(
                shape=polygonShape(box=(config.LEG_W, config.LEG_H)),
                density=75,
                friction=1,
                restitution=0.0,
                categoryBits=0x0020,
                maskBits=0x001)
        )
        leg.ground_contact = False
        leg.color1 = (1, 1, 1)
        leg.color2 = (0.35, 0.35, 0.35)
        rjd = revoluteJointDef(
            bodyA=booster,
            bodyB=leg,
            localAnchorA=(0, 0),
            localAnchorB=(i * config.LEG_AWAY, config.LEG_DOWN),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=config.LEG_SPRING_TORQUE,
            motorSpeed=+0.1 * i  # low enough not to jump back into the sky
        )
        if i == -1:
            rjd.lowerAngle = +0.4  # Yes, the most esoteric numbers here, angles legs have freedom to travel within
            rjd.upperAngle = +0.65
        else:
            rjd.lowerAngle = -0.65
            rjd.upperAngle = -0.4
        leg.joint = world.CreateJoint(rjd)
        legs.append(leg)

    return legs

def generate_particle(world, x, y, mass):
    p = world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=0.5, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3)
        )

    return p
