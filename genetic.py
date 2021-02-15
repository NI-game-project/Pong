import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import random
import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense
from keras.optimizers import Adam, RMSprop

import os
import agent_hyper as trainer
import networks
import gym
import logger 

class Genetic_Algorithm():

    def __init__(self, population_size, elite_workers_num, entropy_num, train_episode_num, generations, epsilon, env_name, save_every, network_type, log_dir, seed, lr):
        
        self.log_dir = log_dir

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.training = True
        self.random = True
        self.load_data = False
        self.zero_fixer = 1e-9
        self.fan_in = 6400 *4
        self.save_every = save_every
        self.type = network_type
        
        self.train_episode_num = train_episode_num
        self.population_size = population_size
        self.elite_workers_num = elite_workers_num
        self.generations = generations
        self.epsilon = epsilon
        self.entropy_elite_workers_num = entropy_num

        self.lr = lr
        self.batch_size = 1 
        self.lamBda = 1
        self.episodes = 1
        self.decay_rate = 0.9999
        
        self.logger = logger.Logger(self.log_dir)
        self.workers = []
        
        self.env_name = env_name 
        self.setup = 'genetic'
        self.seed = seed

        self.agent = trainer.A2C_Agent(self.env_name, self.log_dir, self.setup, self.lr, self.batch_size, self.lamBda, self.episodes, self.decay_rate, self.save_every)

        self.weight_space = 225_073 + 226_358 # Just holds for A2C
        
    def set_parameters_dqn(self, worker):
        
        for k, x in enumerate(worker):
            np.random.seed(x)
            if k == 0:
                weights = np.random.uniform(low=-np.sqrt(6/self.fan_in), high=np.sqrt(6/self.fan_in), size=self.weight_space)
            else:
                weights += self.epsilon * np.random.normal(0,1,size=self.weight_space)

        last_used = 0

        for i in range(len(self.agent.q_estimator.layers)):

            if 'dense' in self.agent.q_estimator.layers[i].name:
                weights_shape = self.agent.q_estimator.layers[i].kernel.shape
                no_of_weights = tf.reduce_prod(weights_shape)
                new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
                last_used += no_of_weights
                
                weights_shape_bias = self.agent.q_estimator.layers[i].bias.shape
                no_of_weights_bias = tf.reduce_prod(weights_shape_bias)
                new_weights_bias = tf.reshape(weights[last_used:last_used+no_of_weights_bias], weights_shape_bias) 
                
                self.agent.q_estimator.layer[i].set_weigths([new_weights, new_weights_bias])
                last_used += no_of_weights_bias

        self.agent.target_estimator.set_weights(self.agent.q_estimator.get_weights())

    def set_parameters_a2c(self, worker):

        for k, x in enumerate(worker):
            np.random.seed(x)
            if k == 0:
                weights = np.random.uniform(low=-np.sqrt(6/self.fan_in), high=np.sqrt(6/self.fan_in), size=self.weight_space)
            else:
                weights += self.epsilon * np.random.normal(0,1,size=self.weight_space)
    
        last_used = 0
                
        for i in range(len(self.agent.Actor.layers)):

            if 'conv' in self.agent.Actor.layers[i].name or  'dense' in self.agent.Actor.layers[i].name:
                weights_shape = self.agent.Actor.layers[i].kernel.shape
                no_of_weights = tf.reduce_prod(weights_shape)
                new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
                last_used += no_of_weights
                
                weights_shape_bias = self.agent.Actor.layers[i].bias.shape
                no_of_weights_bias = tf.reduce_prod(weights_shape_bias)
                new_weights_bias = tf.reshape(weights[last_used:last_used+no_of_weights_bias], weights_shape_bias) 
                
                self.agent.Actor.layers[i].set_weights([new_weights,new_weights_bias])
                last_used += no_of_weights_bias

        for i in range(len(self.agent.Critic.layers)):

            if 'conv' in self.agent.Critic.layers[i].name or  'dense' in self.agent.Critic.layers[i].name:
                weights_shape = self.agent.Critic.layers[i].kernel.shape
                no_of_weights = tf.reduce_prod(weights_shape)
                new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
                last_used += no_of_weights
                
                weights_shape_bias = self.agent.Critic.layers[i].bias.shape
                no_of_weights_bias = tf.reduce_prod(weights_shape_bias)
                new_weights_bias = tf.reshape(weights[last_used:last_used+no_of_weights_bias], weights_shape_bias) 
                
                self.agent.Critic.layers[i].set_weights([new_weights,new_weights_bias])
                last_used += no_of_weights_bias
        
    def initial_population(self):

        for _ in range(self.population_size):

            z = random.randint(0,1_000_000)
            self.workers.append([z])
        
    def evaluate_population(self):

        elite_workers = []
        rewards = []

        for worker in self.workers:
            
            if self.type == 'dqn':
                self.set_parameters_dqn(worker)
            elif self.type == 'a2c': 
                self.set_parameters_a2c(worker)

            if self.training == True:
                reward = 0 
                for _ in range(self.train_episode_num):
                    self.agent.episodes = 1
                    self.loss_actor, self.loss_critic, self.entropy_loss, self.predictions, payoff = self.agent.normal_run()
                    reward += payoff
            rewards.append(reward)
        
        # For Pong the rewards are negative, so the order has to be changed
        rewards = np.array(rewards)
        elite_idx = np.argsort(rewards)[len(rewards)-self.elite_workers_num:]

        for idx in elite_idx:
            elite_workers.append(self.workers[idx])
        
        elite_workers = self.evaluate_diversity_weights(elite_workers)

        return elite_workers, rewards

    def evaluate_diversity_actions(self,elite_workers):
        pass

    
    def evaluate_diversity_weights(self, elite_workers):

        networks = []
        self.entropy_score = []
        input_size = self.agent.mini_batch_size
        noise_batch_size = self.agent.mini_batch_size
        
        for worker in self.workers:
            for k, x in enumerate(worker):
                np.random.seed(x)
                if k == 0:
                    weights = np.random.uniform(low=-np.sqrt(6/self.fan_in), high=np.sqrt(6/self.fan_in), size=self.weight_space)
                else:
                    weights += self.epsilon * np.random.normal(0,1,size=self.weight_space)
        
            networks.append(weights)

        for idx in range(len(self.workers)):

            flattened_network = networks.copy()
            flattened_network.pop(idx)
            flattened_network = np.vstack(flattened_network).astype('float32')
            mutual_distances = tf.math.reduce_sum(tf.math.abs(tf.expand_dims(flattened_network, 0) - tf.expand_dims(flattened_network, 1)), 2,name='mutual_squared_distances') # all distances between weight vector samples
            nearest_distances = tf.identity(-1*tf.math.top_k(-1 * mutual_distances, k=2)[0][:, 1] ,name='nearest_distances') # distance to nearest neighboor for each weight vector sample
            entropy_estimate = tf.identity(input_size * tf.math.reduce_mean(tf.math.log(nearest_distances)) + tf.math.digamma(tf.cast(noise_batch_size, tf.float32)), name='entropy_estimate')
            self.entropy_score.append(entropy_estimate)

        elite_idx = np.argsort(self.entropy_score)[:self.entropy_elite_workers_num]

        for idx in elite_idx:
            elite_workers.append(self.workers[idx])

        return elite_workers


    def mutate_population(self, elite_workers):

        self.workers = []
        self.workers.append(elite_workers[self.elite_workers_num])
        self.elite = elite_workers[0]

        for _ in range(self.population_size - 1):
            
            idx = random.randint(1,self.elite_workers_num)
            seed = random.randint(0,1_000_000)
            new_worker = elite_workers[idx].copy()
            new_worker.append(seed)
            self.workers.append(new_worker) 
        
    def run(self):
        
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.initial_population()
        max_reward = 0

        for step in range(self.generations):
            
            elite_workers, rewards = self.evaluate_population()
            
            self.mutate_population(elite_workers)
            
            if step % 5 == 0:
                    self.logger.log_performance(step, tf.reduce_mean(rewards).numpy(), self.loss_actor.numpy(), self.loss_critic.numpy(), self.entropy_loss.numpy(), tf.reduce_mean(self.entropy_score).numpy(), 0, 0, self.predictions)
                
            if step % self.save_every == 0:

                max_reward = tf.reduce_mean(rewards).numpy()

                path = '{}/{:.3f}.txt'.format(self.log_dir,max_reward)
                with open(path, 'w') as data:
                    for x in elite_workers:
                        data.write('{}\n'.format(x))
                data.close()

                
                path_a = '{}/actor_{:.3f}.h5'.format(self.log_dir,max_reward)
                path_c = '{}/critic_{:.3f}.h5'.format(self.log_dir,max_reward)


                if self.type == 'a2c':
                    self.agent.Critic.save(path_c)
                    self.agent.Actor.save(path_a)
                else:
                    self.agent.q_estimator.save(path)
            
            elite_workers, rewards = [],[]



