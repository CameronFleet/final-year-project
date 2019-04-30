import matplotlib.pyplot as plt
import numpy as np

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

def smoothed_plot(x, rewards, window_size = 50, label=""):
    
    y = []
    for i in range(len(rewards)):
    
        low  = int(i-(window_size/2) if i-(window_size/2) > 0 else 0)
        high = int(i+(window_size/2) if i+(window_size/2) < len(rewards) else len(rewards))
        y.append(sum([r for r in rewards[low:high]]) / (high-low))
    
    
    plt.plot(x, y, label=label)


def average_graph(files):

    x = []
    y = []

    for file in files[0]: 
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
                if files[1] == "DDQN":
                    y[i] += values[i] + 60
                else:
                    y[i] += values[i]
        
    for i  in range(len(y)):
        y[i] = y[i]/len(files[0])

    smoothed_plot(x,y,20,files[1])

def merged_plot(data):

    for files in data:
        average_graph(files)

    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.title("SGD Sarsa trained on NoisyBoosterLander with different learning rates")
    plt.legend()
    plt.show()



def read_episode_save(file, metric):

    f = open(file, 'r')

    for line in f:
        line = line.split(":")
        line_metric = line[1].split(",")[0]

        print(line_metric)
        if line_metric == metric:

            line_data = line[2]
            line_data = line_data[:-3]

            data = []

            for val in line_data.split(","):
                data.append(float(val))

            return data

def plot_metric_from_save(pid, metric, oscil, damp, oscil_label, damp_label):
        
    y_oscil = read_episode_save("/Users/cameronfleet/Desktop/University/PROJECT/pid tuning/tuning_{}/{}/episode.save".format(pid,oscil), metric)
    y_damp = read_episode_save("/Users/cameronfleet/Desktop/University/PROJECT/pid tuning/tuning_{}/{}/episode.save".format(pid,damp), metric)


    plt.plot(np.arange(len(y_oscil)), y_oscil, label=oscil_label)
    plt.plot(np.arange(len(y_damp)), y_damp, label=damp_label)

    plt.xlabel("timestep")
    plt.ylabel(metric)
    plt.title("{} agiasnt time ".format(metric))
    plt.legend(loc="upper right")
    plt.show()

if __name__ == '__main__':

    first = ([], "Q-learning SGD")
    for i in range(20):
        first[0].append('/Users/cameronfleet/Desktop/eval/bl_noisy_sgd/q_sgd_005/{}.txt'.format(i))

    second = ([], "Sarsa SGD")
    for i in range(20):
        second[0].append('/Users/cameronfleet/Desktop/eval/bl_noisy_sgd/sarsa_sgd_001/{}.txt'.format(i))

    third = ([], "DDQN")
    for i in range(20):
        third[0].append('/Users/cameronfleet/Desktop/eval/bl_noisy_sgd/sarsa_sgd_0005/{}.txt'.format(i))
   
    fourth = ([], "DDQN")
    for i in range(20):
        fourth[0].append('/Users/cameronfleet/Desktop/eval/bl_noisy_new_reward/{}.txt'.format(i))

    data = [first, second, fourth]
    merged_plot(data)

    # plot_metric_from_save(  pid="angular",
    #                         metric="Theta (rad)", 
    #                         oscil="episode_249_second_oscil", 
    #                         damp="episode_264_second_damped", 
    #                         oscil_label="K_p=15", 
    #                         damp_label="K_d=9.35")




