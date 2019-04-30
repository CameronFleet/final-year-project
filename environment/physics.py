import math

def drag_force(body, air_density, drag_constant):
    vel = body.linearVelocity
    cog = body.worldCenter
    Ay = body.diameter * (abs(body.diameter*math.cos(body.angle)) 
                                    + abs(body.height*math.sin(body.angle)))
    Ax = body.diameter * (abs(body.diameter*math.sin(body.angle)) 
                                    + abs(body.height*math.cos(body.angle)))

    drag = (drag_constant*air_density*vel.x*vel.x*Ax) / 2, (drag_constant*air_density*vel.y*vel.y*Ay) / 2
    return drag, cog

def impulse(body):
    vel = body.linearVelocity
    mass = body.mass
    return vel.length*mass  

def fuel_usage(t, mass_flow_rate, avg_utilisation):
    return mass_flow_rate*t*avg_utilisation

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