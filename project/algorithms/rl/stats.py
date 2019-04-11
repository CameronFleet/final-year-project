import matplotlib.pyplot as plt
from collections import deque

class Stats:
    
    def __init__(self, max_ep):
        self.rewards = []
        self.epsilons = []
        self.max_ep = max_ep
        self.moving_avg = deque(maxlen=100)
        self.best_performing = 0
        
    def record(self, epsilon):
        self.rewards.append(0)
        self.epsilons.append(epsilon)

    def update(self, ep, reward):
        self.rewards[ep] = self.rewards[ep]  + reward

    def episode_end(self, early_stopping):
        last_reward = self.rewards[-1]
        last_epsilon = self.epsilons[-1]
        self.moving_avg.append(last_reward)
        moving_avg = sum(self.moving_avg)/ 100
        print("Episode {}/{} Reward={} Epsilon={} 100avg={}".format(len(self.rewards), self.max_ep, last_reward, last_epsilon, moving_avg))
        
        if moving_avg > early_stopping and moving_avg > self.best_performing:
            self.best_performing = moving_avg
            return moving_avg
        else: 
            return False

    def plot(self,  window_size = 100, show=False, title="",):
        
        plt.figure()
        y = []
        for i in range(len(self.rewards)):
            
            low  = int(i-(window_size/2) if i-(window_size/2) > 0 else 0)
            high = int(i+(window_size/2) if i+(window_size/2) < len(self.rewards) else len(self.rewards))
            y.append(sum([r for r in self.rewards[low:high]]) / (high-low))
        
        
        plt.plot([ ep for ep in range(len(self.rewards))], y)
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.title(title)
        if show:
            plt.show()

    def save_progress(self, title, window_size, path):
        self.plot(window_size, title=title)
        plt.savefig(path)