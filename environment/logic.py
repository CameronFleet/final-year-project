from environment.physics import impulse
import environment.config as config

def episode_complete(env):

    booster = env.booster
    legs    = env.legs
    done = False
    landed = None
    imp = None
    reward = None
    vel = booster.body.linearVelocity
    pos = booster.body.position
    
    # Hits the ground
    if env.game_over:
        done = True
        landed = False
        reward = -50 - abs(vel.length)
        imp = impulse(booster.body)

    # Goes off screen
    if pos.x < -50 or pos.x > config.WORLD_W + 50 or pos.y < -50 or pos.y > config.WORLD_H + 50:
        done = True
        landed = False
        reward = -50

    # Touchdown
    if legs[0].ground_contact and legs[1].ground_contact :
        done = True
        landed = True
        reward = +200 - 5*abs(vel.length)
        imp = impulse(booster.body)

        # Was 50 - abs(vel)
        # Was 100 -3abs(vel)

    # Manual termination    
    if env.done: 
        done = True
        landed = False
        reward = 0

    # Running for too long!
    if env.steps > env.termination_time and env.time_terminated:
        reward = -100 #Was 50
        landed = False
        done = True

    return done, landed, imp, reward
