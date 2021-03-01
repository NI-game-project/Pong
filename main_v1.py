from agent_hyper import A2C_Agent
from genetic import Genetic_Algorithm
from a3c import A3C_Agent


########## RUN NORMAL ##########      

#################
save_every = 5000
'''
episodes = 10_001
batch_size = 4
lr = 3e-4
decay_rate = 0.99998
lamBda = 10_000
save_path = 'experiments/normal/v1'
env_name = 'PongDeterministic-v4'
setup = 'normal'
agent = A2C_Agent(env_name, save_path, setup, lr, batch_size, lamBda, episodes, decay_rate, save_every)

agent.normal_run()

agent.logger.close_files()
agent.logger.plot('Normal V1')
'''

########## RUN HYPERNETWORK 1 ########

episodes = 10_001
batch_size = 4
lr = 1e-4
decay_rate = 0.9999
lamBda = 5e-6
save_path = 'experiments/hypernetwork/v1'
env_name = 'PongDeterministic-v4'# 'PongNoFrameskip-v4'
setup = 'hypernetwork'
agent = A2C_Agent('name', env_name, save_path, setup, lr, batch_size, lamBda, episodes, decay_rate, save_every)

agent.run_hypernetwork()
#agent.train(4)

agent.logger.close_files()
agent.logger.plot('Hypernetwork V1')

########## RUN HYPERNETWORK 2 ##########


episodes = 3_001
batch_size = 4
lr = 5e-4
decay_rate = 0.9999
lamBda = 10_000
save_path = 'experiments/hypernetwork/v2'
env_name = 'PongDeterministic-v4'
setup = 'hypernetwork'
agent = A2C_Agent(env_name, save_path, setup, lr, batch_size, lamBda, episodes, decay_rate, save_every)

agent.run_hypernetwork()

agent.logger.close_files()
agent.logger.plot('Hypernetwork V2')

########## RUN HYPERNETWORK 3 ##########

episodes = 3_001
batch_size = 4
lr = 1e-3
decay_rate = 0.9999
lamBda = 10_000
save_path = 'experiments/hypernetwork/v3'
env_name = 'PongDeterministic-v4'
setup = 'hypernetwork'
agent = A2C_Agent(env_name, save_path, setup, lr, batch_size, lamBda, episodes, decay_rate, save_every)

agent.run_hypernetwork()

agent.logger.close_files()
agent.logger.plot('Hypernetwork V3')

########## RUN HYPERNETWORK 3 ##########

episodes = 3_001
batch_size = 4
lr = 5e-6
decay_rate = 0.9999
lamBda = 10_000
save_path = 'experiments/hypernetwork/v4'
env_name = 'PongDeterministic-v4'
setup = 'hypernetwork'
agent = A2C_Agent(env_name, save_path, setup, lr, batch_size, lamBda, episodes, decay_rate, save_every)

agent.run_hypernetwork()

agent.logger.close_files()
agent.logger.plot('Hypernetwork V4')


########## RUN GENETIC ##########

population_size = 32
generations = 301
epsilon = 0.002
train_episode_num = 3
elite_workers_num = 8
entropy_num = 4
env_name = 'PongDeterministic-v4'
lr = 3e-4
seed = 42
network_type = 'a2c'        
log_dir = 'experiments/genetic/v1'

##############
save_every = 50

ga = Genetic_Algorithm(population_size, elite_workers_num, entropy_num, train_episode_num, generations, epsilon, env_name, save_every, network_type, log_dir, seed, lr)
ga.run()

ga.logger.close_files()
ga.logger.plot('Genetic V1')

