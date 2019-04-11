import matplotlib.pyplot as plt

class Stats:
    
    def __init__(self, max_ep):
        self.rewards = []
        self.epsilons = []
        self.max_ep = max_ep
        
    def record(self, epsilon):
        self.rewards.append(0)
        self.epsilons.append(epsilon)

    def update(self, ep, reward):
        self.rewards[ep] = self.rewards[ep]  + reward;

    def show(self):
        last_reward = self.rewards[-1]
        last_epsilon = self.epsilons[-1]
        print("Episode {}/{}. Reward {}. Epsilon {}.".format(len(self.rewards), self.max_ep, last_reward, last_epsilon))
        
    def plot(self, window_size = 20):
        
        y = []
        for i in range(len(self.rewards)):
            
            low  = int(i-(window_size/2) if i-(window_size/2) > 0 else 0)
            high = int(i+(window_size/2) if i+(window_size/2) < len(self.rewards) else len(self.rewards))
            y.append(sum([r for r in self.rewards[low:high]]) / (high-low))
        
        
        plt.plot([ ep for ep in range(len(self.rewards))], y)
        plt.xlabel("episode")
        plt.ylabel("reward")

    def save_progress(self, window_size, path):
        self.plot(10)
        plt.savefig(path)
