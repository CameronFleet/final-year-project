import sys
sys.path.append("/Users/cameronfleet/Desktop/University/PROJECT/dev/")

from algorithms.pid.controller import Controller
from algorithms.pid.controller import PID
def record_episode(seed):
    f = open("pid/save.log")
    episode_number = len(f.readlines())
    f = open("pid/save.log", "a")
    f.write("EPISODE="+ str(episode_number) + " SEED=" + str(seed) + "\n")
    return episode_number