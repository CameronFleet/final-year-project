
from statistics import mode
import numpy as np
import math
import matplotlib.pyplot as plt
    
class PID: 

    def __init__(self, timestep, consts = (1,1,1)):
        self.prev_error = None
        self.integral   = 0
        self.result     = 0
        self.consts     = consts
        self.time_step  = timestep

    def calculate(self, e, de=None, bias = 0):

        K = self.consts
        self.integral = self.integral + (e* self.time_step)

        if de is None:
            if self.prev_error is not None:
                de = (e - self.prev_error)/self.time_step
            else:
                de = 0

        self.prev_error = e
        result = K[0]*e + K[1]*self.integral + K[2]*de + bias
        return result

class PIDAlg:

    def __init__(self, timestep):
        self.time_step   = timestep
        self.altitude_pid   = PID(self.time_step,(-0.24,0,0.5))
        self.x_pid          = PID(self.time_step,(0.005625,0.028125,0.0075))
        self.angular_pid    = PID(self.time_step,(25,0,0.1))
        self.x_metrics        = [(0,0,0)]
        self.altitude_metrics = [(0,0,0)]
        self.angular_metrics = [(0,0,0)]

        self.times         = [0] 

    def go(self, state, env): 

        # State feedback
        x, y, vx, vy,theta, vtheta, alpha, l1, l2 = state

        # FT PID
        Ft = self.altitude_pid.calculate(y-env.GOAL[1], de=-vy)
        Ft = 0 if  Ft < 0 else Ft
        Ft = 1 if Ft > 1 else Ft
        if l1 or l2:
            Ft=0
        self.altitude_metrics.append((Ft, y, vy))
        
        # ALPHA PID
        alpha = self.x_pid.calculate(env.GOAL[0]-x, de=-vx)
        alpha = -0.5 if alpha < -0.5 else alpha
        alpha = 0.5 if alpha > 0.5 else alpha
        self.x_metrics.append((alpha, x, vx))

        # ANGULAR PID
        Fs = -self.angular_pid.calculate(theta, de=vtheta)
        Fs = -1 if Fs < -1 else Fs
        Fs = 1 if Fs > 1 else Fs
        self.angular_metrics.append((Fs, theta, vtheta))

        # Actions selected for next world step
        actions=[Ft, 0 , Fs]

        self.times.append(self.times[-1] + self.time_step)
        return actions

    def report(self):
        K = self.altitude_pid.consts
        plt.figure(0)
        plt.title("Ft vs Time for Kp,Ki,Kd = " + str(K[0])+ ", " +str(K[1])+ ", "+ str(K[2]))
        plt.plot(self.times[1:], [ m[0] for m in self.altitude_metrics[1:]], label="Control Signal")
        plt.legend()

        plt.figure(1)
        plt.title("Altitude and Vertical Velocity vs Time")
        plt.plot(self.times[1:], [ m[1] for m in self.altitude_metrics[1:]], label="Altitude")
        plt.plot(self.times[1:], [ m[2] for m in self.altitude_metrics[1:]], label="Vertical Velocity")
        plt.legend()

        K = self.x_pid.consts
        plt.figure(2)
        plt.ylim(-.51,.51)
        plt.title("Alpha vs Time for Kp,Ki,Kd = " + str(K[0])+ ", " +str(K[1])+ ", "+ str(K[2]))
        plt.plot(self.times[1:], [ m[0] for m in self.x_metrics[1:]], label="Control Signal")

        K = self.angular_pid.consts        
        plt.figure(3)
        plt.ylim(-1,1)
        plt.title("Fs vs Time for Kp,Ki,Kd = " + str(K[0])+ ", " +str(K[1])+ ", "+ str(K[2]))
        plt.plot(self.times[1:], [ m[0] for m in self.angular_metrics[1:]], label="Control Signal")

        plt.show()


