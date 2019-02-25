
from statistics import mode
import numpy as np
import math

# class Trajectory:

#     def __init__(self): 
#         self.initated_landing = False

#     def calculate(self, s, env, step):
#         x, y, vx, vy,theta, vtheta, alpha, l1, l2= s
        
#         # Resolved forces vertically, using F = ma => a = F/m
#         ay = (env.agent.MAIN_ENGINE_POWER*math.cos(theta + alpha)*106.5 + env.agent.body.mass*env.world.gravity[1] ) / env.agent.body.mass
        
#         # Calculate distance required to perform hoverslam using v^2 = u^2 + 2as
#         miniumum_stopping_distance =  ( (math.pow(vy,2)) / (2*ay) )

#         if step % 20 == 0:
#             print("AY: ", ay)
#             print("Vy: ", vy)
#             print("MIN_STOP_DISTANCE: ", miniumum_stopping_distance, "y: ", y)

#         if miniumum_stopping_distance > y or self.initated_landing:
#             self.initated_landing = True
#             # Goal is to come to a standstill - straight down
#             return [0,0,0]
#         else:
#             # Calculate angle of approach
#             descent_angle = math.atan( (env.GOAL[0] - x) / (y - miniumum_stopping_distance) )
#             if step % 20 == 0:
#                 print("DESCENT: ", descent_angle)
#             # Keep vy as it is - ensure vx is enough to match velocity
#             # return [vy*math.tan(descent_angle), vy, descent_angle]
#             return None
    
class PID: 

    def __init__(self, timestep, consts = (1,1,1)):
        self.prev_error = 0
        self.integral   = 0
        self.result     = 0
        self.consts     = consts
        self.time_step  = timestep

    def calculate(self, e, de=None, bias = 0):

        K = self.consts
        self.integral = self.integral + (e* self.time_step)

        if de is None:
            de = (e - self.prev_error)/self.time_step

        self.prev_error = e
        result = K[0]*e + K[1]*self.integral + K[2]*de + bias
        
        return result
        
class PIDAlg:

    def __init__(self, timestep):
        self.time_step   = timestep
        self.error_margin = 0.0

    def objective(x_error, y_error, angle_error):
        return x_error + y_error + angle_error

    def _result_function(self, r, actions, margin = 0):
        if r > margin:
            return actions[0]
        elif r < -margin:
            return actions[1]
        else: 
            return 0

    def _sigmoid(self,x):
        return 1 / (1 + math.exp(-x))

    def go(self, env, seed=None, render=False): 

        actions = []
        env.seed(seed)
        step = 0 
        y_vel_pid = PID(self.time_step,(1,0,1.5))

        print("START OF EPISODE ===== ")
        print("GOAL: ", env.GOAL[0], "  , ", env.GOAL[1])

        while True: 
            # Step through world
            step += 1

            # Plant equation 
            s, r, done, _ = env.step(actions)

            # State feedback
            x, y, vx, vy,theta, vtheta, alpha, l1, l2 = s

            # Controller generates FT
            Ft = 1 if y_vel_pid.calculate(y-env.GOAL[1]) < 0 else 0
            print(Ft)

            # Actions selected for next world step
            actions=[Ft, 0 , 0]

            if l1 or l2:
                actions = []

            if render:
                env.render()

            if done: 
                actions = []

        return True


