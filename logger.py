import os
import csv
import matplotlib.pyplot as plt


class Logger(object):

    def __init__(self, log_dir):
        
        self.log_dir = log_dir
        self.txt_path = os.path.join(log_dir, 'log.txt')
        self.csv_path = os.path.join(log_dir, 'performance.csv')
        self.fig_path = os.path.join(log_dir, 'fig.png')

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.txt_file = open(self.txt_path, 'w')
        self.csv_file = open(self.csv_path, 'w')
        
        fieldnames = ['generation', 'reward', 'actor_loss', 'critic_loss', 'entropy_loss', 'diversity', 'loss', 'learning_rate', 'actions']
        
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

    def log(self, text):
        
        self.txt_file.write(text+'\n')
        self.txt_file.flush()
        print(text)

    def log_performance(self, generation, reward, actor_loss, critic_loss, entropy_loss, diversity, loss, learning_rate, actions):
        ''' Log a point in the curve
        Args:
            timestep (int): the timestep of the current point
            reward (float): the reward of the current point
        '''
        self.writer.writerow({'generation': generation,
                             'reward': reward,
                             'actor_loss': actor_loss,
                             'critic_loss': critic_loss,
                             'entropy_loss': entropy_loss,
                             'diversity': diversity,
                             'loss': loss,
                             'learning_rate': learning_rate,
                             'actions': actions})
        print('')
        self.log('----------------------------------------')
        self.log('  generation   |  ' + str(generation))
        self.log('  reward       |  ' + str(reward))
        self.log('  actor loss   |  ' + str(actor_loss))
        self.log('  criic loss   |  ' + str(critic_loss))
        self.log('  entropy loss |  ' + str(entropy_loss))
        self.log('  diversity    |  ' + str(diversity))
        self.log('  loss         |  ' + str(loss))
        self.log('  learningrate |  ' + str(learning_rate))
        self.log('  actions      |  ' + str(actions))
        self.log('----------------------------------------')

    def plot(self, algorithm):

        plot(self.csv_path, self.fig_path, algorithm)

    def close_files(self):
        
        if self.txt_path is not None:
            self.txt_file.close()
        if self.csv_path is not None:
            self.csv_file.close()

def plot(csv_path, save_path, algorithm):
    
    with open(csv_path) as csvfile:
        print(csv_path)
        reader = csv.DictReader(csvfile)
        xs = []
        ys1 = []
        ys2 = []
        for row in reader:
            xs.append(int(row['generation']))
            ys1.append(float(row['reward']))
            ys2.append(float(row['loss']))

        fig, ax = plt.subplots()
        ax2 = ax.twinx()

        ax.plot(xs, ys1, label=algorithm)
        ax2.plot(xs, ys2, label=algorithm)
        ax.set(xlabel='generation', ylabel='reward')
        ax2.set(xlabel='generation', ylabel='loss')
        ax.legend()
        ax2.legend()
        ax.grid()
        ax2.grid()

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig.savefig(save_path)
