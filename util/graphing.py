import matplotlib.pyplot as plt

def read_file(filename, key_pos = 0, value_pos = 1):
    f = open(filename, 'r')

    keys = []
    values = []
    for line in f:
        row = line.split(',')
        keys.append(float(row[key_pos]))
        values.append(float(row[value_pos]))

    f.close()
    return keys, values

def smoothed_plot(x, rewards, window_size = 50):
    
    y = []
    for i in range(len(rewards)):
    
        low  = int(i-(window_size/2) if i-(window_size/2) > 0 else 0)
        high = int(i+(window_size/2) if i+(window_size/2) < len(rewards) else len(rewards))
        y.append(sum([r for r in rewards[low:high]]) / (high-low))
    
    
    plt.plot(x, y)
    plt.xlabel("episodes")
    plt.ylabel("reward")
    plt.show()

def average_graph(files):

    x = []
    y = []

    for file in files: 
        keys, values = read_file(file)

        if x == []:
            x = keys
        
        if len(keys) < len(x):
            x = keys
            y = y[0:len(x)]

        if y == []:
            y = values
        else:
            for i in range(len(y)):
                y[i] += values[i]
        
    for i  in range(len(y)):
        y[i] = y[i]/len(files)

    smoothed_plot(x,y,20)

if __name__ == '__main__':
    files = []

    for i in range(20):
        files.append('/Users/cameronfleet/Desktop/eval/bl_broken_better_reward_2/bbl_acc0.5_gps0_rate0_thrust0/{}.txt'.format(i))

    average_graph(files)