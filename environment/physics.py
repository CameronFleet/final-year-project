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