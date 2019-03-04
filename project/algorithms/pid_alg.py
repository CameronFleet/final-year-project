
from statistics import mode
import numpy as np
import math
try:
    import matplotlib.pyplot as plt
except:
    pass
import os
    
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

    def __init__(self, timestep, seed, episode_number):
        self.time_step   = timestep
        self.altitude_pid   = PID(self.time_step,(-0.24,0,0.5))
        self.x_pid          = PID(self.time_step,(0.005625,0.028125,0.0075))
        self.angular_pid    = PID(self.time_step,(25,0,0.1))
        self.times         = [0] 
        self.seed          = seed
        self.episode_number = episode_number

        self.metrics = {"Horizontal_Displacement (m)":[], 
                        "Altitude (m)": [], 
                        "Vertical_Velocity (m/s)":[],
                        "Horizontal_Velocity (m/s)":[],
                        "Theta (rad)":[], 
                        "Angular_Velocity (rad/s)":[], }
        self.control_metrics = {"Thrust":[], 
                                "Alpha":[],
                                "Side Thrust":[]}

    def _record_metrics(self, metrics, isControl=False):

        if isControl:
            for key, value in metrics.items():
                self.control_metrics[key].append(value)
        else:
            for key, value in metrics.items():
                self.metrics[key].append(value)

    def go(self, state, env): 

        # State feedback
        x, y, vx, vy,theta, vtheta, alpha, l1, l2 = state

        # FT PID
        Ft = self.altitude_pid.calculate(y-env.GOAL[1], de=-vy)
        Ft = 0 if  Ft < 0 else Ft
        Ft = 1 if Ft > 1 else Ft
        if l1 or l2:
            Ft=0
        
        # ALPHA PID
        alpha = self.x_pid.calculate(env.GOAL[0]-x, de=-vx)
        alpha = -0.5 if alpha < -0.5 else alpha
        alpha = 0.5 if alpha > 0.5 else alpha

        # ANGULAR PID
        Fs = -self.angular_pid.calculate(theta, de=vtheta)
        Fs = -1 if Fs < -1 else Fs
        Fs = 1 if Fs > 1 else Fs

        # Actions selected for next world step
        actions=[Ft, 0 , Fs]

        self._record_metrics({  "Thrust":Ft*100, 
                                "Alpha":alpha,
                                "Side Thrust":Fs*100}, 
                                True)

        self._record_metrics({  "Horizontal_Displacement (m)":x, 
                                "Altitude (m)": y, 
                                "Vertical_Velocity (m/s)":vy,
                                "Horizontal_Velocity (m/s)":vx,
                                "Theta (rad)":theta, 
                                "Angular_Velocity (rad/s)":vtheta,
                            })
        self.times.append(self.times[-1] + self.time_step)
        return actions

    def report(self, save=True, onlyControl=False):

        if save:
            os.system("mkdir episodes/episode_"+str(self.episode_number))
            os.system("touch episodes/episode_"+str(self.episode_number)+"/episode.save")

        lims   = {  "Thrust":(0,100,10), 
                    "Alpha":(-0.5, 0.5,0.1), 
                    "Side Thrust":(-100,100,10)}
        consts = {  "Thrust":self.altitude_pid.consts, 
                    "Alpha":self.x_pid.consts, 
                    "Side Thrust":self.angular_pid.consts}

        units  = { "Thrust":"% of MAX", "Alpha":"rad", "Side Thrust":"% of MAX" } 
        

        if not onlyControl:
            for metric, values in self.metrics.items():
                self._draw_metric(metric, values, save=save)

        for metric, values in self.control_metrics.items():
            self._draw_metric(metric, values, 
                                lim=lims[metric], 
                                isControlSignal=True,
                                K=consts[metric],
                                save=save,
                                unit=units[metric])
        plt.show()

    def _save_metric(self, metric, values):
        f = open("episodes/episode_"+str(self.episode_number)+"/episode.save", "a")
        f.write("{METRIC:" + metric + ",VALUES:")
        for i, value in enumerate(values):
            if i == len(values) -1:
                f.write(str(value) + "} \n")
            else:
                f.write(str(value) + ",")


    def _draw_metric(self, metric, values, lim=None, isControlSignal=None, K=None, save=False, unit=None):

        if len(values) == 0:
            return 

        plt.figure(metric)
        plt.plot(self.times[1:], values, label=metric)
        plt.xlabel("Time (s)")
        if unit is not None:
            plt.ylabel(metric + " ("+unit+")")
        else:
            plt.ylabel(metric )

        if lim is not None:
                plt.ylim((lim[0]-lim[2], lim[1]+lim[2]))

        if save: 
            self._save_metric(metric, values)

        if isControlSignal:
            plt.title(metric+" vs Time with Kp=" + str(K[0])+ " Ki=" +str(K[1])+ " Kd="+ str(K[2]))
            if save:
                plt.savefig("episodes/episode_"+str(self.episode_number)+"/p_" +metric+ "_Kp=" + str(K[0])+ "_Ki=" +str(K[1])+ "_Kd="+ str(K[2]) +".png")
        else:
            plt.title(metric + " vs Time")
            if save:
                plt.savefig("episodes/episode_"+str(self.episode_number)+"/p_" +metric.split()[0] +".png")

 

        



