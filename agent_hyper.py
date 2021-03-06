import os
import random
import gym
import pylab
import numpy as np
import tensorflow as tf 
import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras import backend as K
import networks
import logger
import itertools
import threading


import time

tf.keras.backend.clear_session()

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'GPUs {gpus}')
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: pass

class A2C_Agent:
    # Actor-Critic Main Optimization Algorithm
    def __init__(self, name, env_name, save_path, setup, lr, batch_size, lamBda, episodes, decay_rate, save_every):
        
        # Initialization
        self.name = name
        self.setup = setup
        self.env_name = env_name       
        self.env = gym.make(env_name)
        self.output_shape = self.env.action_space.n
        
        #hyperperameters
        self.episodes = episodes
        self.batch_size = batch_size
        self.lr = lr
        self.decay_rate = decay_rate
        self.lamBda = lamBda
        self.save_every = save_every
        
        self.mini_batch_size = 32
        self.epochs = 5
        self.gamma = 0.95
        self.seed = 42
        np.random.seed(seed= self.seed)
        tf.random.set_seed(self.seed)
        random.seed(self.seed)
        self.env.seed(self.seed)
        self.zero_fixer = 1e-7
        self.episode = 0
        self.lock = threading.Lock()

        #input shape of the pong enviroment
        self.image_width = 80
        self.image_height = 80
        self.channels = 1

        self.layer1_pool_size = 2
        self.layer2_pool_size = 2
        self.layer1_filter_size = 5
        self.layer1_size = 32
        self.layer1_pool_size = 2
        self.number_of_channels = 1
        self.layer2_filter_size = 5
        self.layer2_size = 16
        self.layer2_pool_size = 2
        self.layer3_size = 8
        self.input_shape = (self.channels, self.image_width, self.image_height)
        self.image_memory = np.zeros(self.input_shape)

        # Instantiate games and plot memory
        self.states, self.actions, self.rewards, self.predictions, self.next_states, self.dones = [], [], [], [], [], []
        self.scores, self.average = [], []

        self.memory = [[] for _ in range(self.batch_size)]

        
        self.save_path = save_path
        if not os.path.exists(self.save_path): os.makedirs(self.save_path)
        self.path = '{}_A2C_{}'.format(self.env_name, self.lr)
        self.Model_name = os.path.join(self.save_path, self.path)

        self.logger = logger.Logger(self.save_path)

        self.hypernetwork = networks.Hypernetwork_PONG('Hypernetwork_Pong')
        self.Actor = networks.Actor(input_shape=self.input_shape, output_shape = self.output_shape, seed=self.seed)  
        self.Critic = networks.Critic(input_shape=self.input_shape, output_shape = 1, seed=self.seed)
        
        #print(self.Actor.summary())
        #print(self.Critic.summary())

        self.lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = self.lr, decay_steps=1, decay_rate=self.decay_rate)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.optimizer_c = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.optimizer_a = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

    def set_weights(self, weights_actor, weights_critic, num):

        last_used = 0
        weights = weights_actor[num]
        for i in range(len(self.Actor.layers)):
            if 'conv' in self.Actor.layers[i].name or  'dense' in self.Actor.layers[i].name: 
                weights_shape = self.Actor.layers[i].kernel.shape
                no_of_weights = tf.reduce_prod(weights_shape)
                new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
                self.Actor.layers[i].kernel = new_weights
                last_used += no_of_weights
                
                if self.Actor.layers[i].use_bias:
                    weights_shape = self.Actor.layers[i].bias.shape
                    no_of_weights = tf.reduce_prod(weights_shape)
                    new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
                    self.Actor.layers[i].bias = new_weights
                    last_used += no_of_weights
        
        last_used = 0
        weights = weights_critic[num]  
        for i in range(len(self.Critic.layers)):
          
            if 'conv' in self.Critic.layers[i].name or  'dense' in self.Critic.layers[i].name: 
                weights_shape = self.Critic.layers[i].kernel.shape
                no_of_weights = tf.reduce_prod(weights_shape)
                new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
                self.Critic.layers[i].kernel = new_weights
                last_used += no_of_weights
                
                if self.Critic.layers[i].use_bias:
                    weights_shape = self.Critic.layers[i].bias.shape
                    no_of_weights = tf.reduce_prod(weights_shape)
                    new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
                    self.Critic.layers[i].bias = new_weights
                    last_used += no_of_weights

        #for i in range(len(self.Actor.layers)):
         #   if 'conv' in self.Actor.layers[i].name or  'dense' in self.Actor.layers[i].name: 
          #      print(self.Actor.layers[i].kernel)
        
        

    def pre_train(self):

        optimizer = tf.keras.optimizers.Adam(lr=1e-4)
        pretrain = 300

        for i in range(pretrain):

            if i % 1000 == 0:

                self.Actor = networks.Actor(input_shape=self.input_shape, output_shape = self.output_shape, seed=i)  
                self.Critic = networks.Critic(input_shape=self.input_shape, output_shape = 1, seed=i)

            with tf.GradientTape() as tape: 

                w1_true = np.concatenate(self.Actor.layers[0].get_weights(),axis = None)
                w2_true = np.concatenate(self.Critic.layers[2].get_weights(),axis = None)

                w3_true = np.concatenate(self.Actor.layers[5].get_weights(),axis = None)
                w4_true = np.concatenate(self.Critic.layers[5].get_weights(),axis = None)

                w5_true = np.concatenate(self.Actor.layers[6].get_weights(),axis = None)
                w6_true = np.concatenate(self.Critic.layers[6].get_weights(),axis = None)

                w7_true = np.concatenate(self.Actor.layers[7].get_weights(),axis = None)
                w8_true = np.concatenate(self.Critic.layers[7].get_weights(),axis = None)                
                
                z = np.random.uniform(low = -1, high = 1, size = [1,300])

                w1, w2, w3, w4, w5, w6, w7, w8 = self.hypernetwork(z,1)

                w1 = tf.reshape(w1, -1)
                w2 = tf.reshape(w2, -1)
                w3 = tf.reshape(w3, -1)
                w4 = tf.reshape(w4, -1)
                w5 = tf.reshape(w5, -1)
                w6 = tf.reshape(w6, -1)
                w7 = tf.reshape(w7, -1)
                w8 = tf.reshape(w8, -1)

                loss_actor = tf.losses.mse(w1_true, w1) + tf.losses.mse(w3_true, w3) + tf.losses.mse(w5_true, w5) + tf.losses.mse(w7_true, w7)
                loss_critic = tf.losses.mse(w2_true, w2) + tf.losses.mse(w4_true, w4) + tf.losses.mse(w6_true, w6) +tf.losses.mse(w8_true, w8)
                
                loss = loss_actor + loss_critic
                #print(loss.numpy())
                grads = tape.gradient(loss, self.hypernetwork.trainable_weights)
                optimizer.apply_gradients(zip(grads,self.hypernetwork.trainable_weights))


    def reshape_weights(self, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8):

        w1_not_gauged = weights_1[:,:,0:-1]
        b1_not_gauged = weights_1[:,:,-1]
        w2_not_gauged = weights_2[:,:,0:-1]
        b2_not_gauged = weights_2[:,:,-1]
        w3_not_gauged = weights_3[:,:,0:-1]
        b3_not_gauged = weights_3[:,:,-1]
        w4_not_gauged = weights_4[:,:,0:-1]
        b4_not_gauged = weights_4[:,:,-1]
        w5_not_gauged = weights_5[:,:,0:-1]
        b5_not_gauged = weights_5[:,:,-1]
        w6_not_gauged = weights_6[:,:,0:-1]
        b6_not_gauged = weights_6[:,-1]
        w7_not_gauged = weights_7[:,:,0:-1]
        b7_not_gauged = weights_7[:,-1]
        w8_not_gauged = weights_8[:,:,0:-1]
        b8_not_gauged = weights_8[:,-1]

        self.w1 = w1_not_gauged
        self.w2 = w2_not_gauged
        self.w3 = w3_not_gauged
        self.w4 = w4_not_gauged
        self.w5 = w5_not_gauged
        self.w6 = w6_not_gauged
        self.w7 = w7_not_gauged
        self.w8 = w8_not_gauged

        self.b1 = b1_not_gauged
        self.b2 = b2_not_gauged
        self.b3 = b3_not_gauged
        self.b4 = b4_not_gauged
        self.b5 = b5_not_gauged
        self.b6 = b6_not_gauged
        self.b7 = b7_not_gauged
        self.b8 = b8_not_gauged
        
        weights_actor = tf.concat(axis=1,values=[tf.reshape(self.w1, (self.batch_size,-1)), tf.reshape(self.b1, (self.batch_size,-1)),\
            tf.reshape(self.w2, (self.batch_size,-1)), tf.reshape(self.b2, (self.batch_size,-1)),\
            tf.reshape(self.w3, (self.batch_size,-1)), tf.reshape(self.b3,(self.batch_size,-1)), \
            tf.reshape(self.w5, (self.batch_size,-1)), tf.reshape(self.b4,(self.batch_size,-1)), \
            tf.reshape(self.w7,(self.batch_size,-1)), tf.reshape(self.b5, (self.batch_size,-1))])
        
        weights_critic = tf.concat(axis=1, values=[tf.reshape(self.w1,(self.batch_size,-1)), tf.reshape(self.b1, (self.batch_size,-1)),\
            tf.reshape(self.w2,(self.batch_size,-1)), tf.reshape(self.b2,(self.batch_size,-1)),
            tf.reshape(self.w4, (self.batch_size,-1)), tf.reshape(self.b3,(self.batch_size,-1)), \
            tf.reshape(self.w6, (self.batch_size,-1)), tf.reshape(self.b4,(self.batch_size,-1)), \
            tf.reshape(self.w8,(self.batch_size,-1)), tf.reshape(self.b6, (self.batch_size,-1))])
        
        return weights_actor, weights_critic

    def run_hypernetwork(self):

        start = time.time()
        self.pre_train()

        z = np.random.uniform(low = -1, high = 1, size = [self.batch_size,300])

        weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8 = self.hypernetwork(z, self.batch_size)
        weights_actor, weights_critic = self.reshape_weights(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8)

        
        for step in range(self.episodes):

            self.clear_memory()

            for thread in range(self.batch_size):
                
                self.score = 0
                self.set_weights(weights_actor, weights_critic, thread)
                self.play_game(thread)

            for e in range(self.epochs):
            
                with tf.GradientTape() as tape:
                    
                    self.step = step

                    self.loss_acc = 0# np.zeros([self.epochs])


                    weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8 = self.hypernetwork(z, self.batch_size)

                    weights_actor, weights_critic = self.reshape_weights(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8)
                
                    
                    '''
                    n_threads = self.batch_size
                    threads = []
                    envs = [gym.make(self.env_name) for i in range(n_threads)]
                    for i, env in enumerate(envs):
                        env.seed(i)

                    for i in range(n_threads):
                        self.set_weights(weights_actor, weights_critic, i)
                        # Create threads
                        threads.append(threading.Thread(target=self.play_game, daemon=True, args=(self, envs[i], i)))

                    for t in threads:
                        
                        t.start()
                        
                    for t in threads:
                        t.join()
                    '''

                    for num in range(self.batch_size):

                        self.set_weights(weights_actor, weights_critic, num)
                        self.update_weights(num)
                            
                    if self.batch_size > 1:

                        zero_fixer = 1e-8
                        input_noise_size = 30
                        noise_batch_size = tf.identity(self.batch_size,name='noise_batch_size') 

                        flattened_network = tf.concat(axis=1,values=[\
                                tf.reshape(self.w1, [noise_batch_size, -1]),tf.reshape(self.b1, [noise_batch_size, -1]),\
                                tf.reshape(self.w2, [noise_batch_size, -1]),tf.reshape(self.b2, [noise_batch_size, -1]),\
                                tf.reshape(self.w3, [noise_batch_size, -1]),tf.reshape(self.b3, [noise_batch_size, -1]),\
                                tf.reshape(self.w4, [noise_batch_size, -1]),tf.reshape(self.b4, [noise_batch_size, -1]),\
                                tf.reshape(self.w5, [noise_batch_size, -1]),tf.reshape(self.b5, [noise_batch_size, -1]),\
                                tf.reshape(self.w6, [noise_batch_size, -1]),tf.reshape(self.b6, [noise_batch_size, -1]),\
                                tf.reshape(self.w7, [noise_batch_size, -1]),tf.reshape(self.b7, [noise_batch_size, -1]),\
                                tf.reshape(self.w8, [noise_batch_size, -1]),tf.reshape(self.b8, [noise_batch_size, -1])])

                        # entropy estimated using  Kozachenko-Leonenko estimator, with l1 distances
                        mutual_distances = tf.math.reduce_sum(tf.math.abs(tf.expand_dims(flattened_network, 0) - tf.expand_dims(flattened_network, 1)), 2,name='mutual_squared_distances') # all distances between weight vector samples
                        nearest_distances = tf.identity(-1*tf.math.top_k(-1 * mutual_distances, k=2)[0][:, 1] ,name='nearest_distances') # distance to nearest neighboor for each weight vector sample
                        entropy_estimate = tf.identity(input_noise_size * tf.math.reduce_mean(tf.math.log(nearest_distances + zero_fixer)) + tf.math.digamma(tf.cast(noise_batch_size, tf.float32)), name='entropy_estimate')
                        loss_div = tf.identity( - 1 * entropy_estimate)
                        loss = self.loss_acc + self.lamBda * loss_div/self.epochs

                    else:
                        loss_div = 0
                        loss = self.loss_acc 

                    grads = tape.gradient(loss, self.hypernetwork.trainable_weights)
                    self.optimizer.apply_gradients(zip(grads, self.hypernetwork.trainable_weights))

            if step % 5 == 0:
                #print(time.time() - start)
                self.logger.log_performance(step, self.score, self.loss_actor.numpy(), self.loss_critic.numpy(), 0, self.predictions[0:5], loss.numpy(), self.optimizer._decayed_lr(tf.float32).numpy(), self.values[0:4])
            
            if step % self.save_every == 0:
                self.hypernetwork.save_weights('{}/{}_hypernetwork.h5'.format(self.save_path, step))

            #print(self.hypernetwork.summary())

    def play_game_2(self, worker, env, thread):        
        
        env.seed(42)
        frame = env.reset()
        state = worker.GetImage(frame)
        
        states, actions, rewards, predictions, advantages, dones, next_states = [], [], [], [], [], [],[]

        done = False
        score = 0

        self.loss_actor = 0
        self.loss_critic = 0 
        self.entropy_loss = 0

        while not done:

            #self.env.render()

            prediction = worker.Actor(state)[0]
            predictions.append(prediction)             
            action = np.random.choice(self.output_shape, p=prediction.numpy())
            next_state, reward, done, _ = env.step(action)
            next_state = worker.GetImage(next_state)

            states.append(state)
            next_states.append(next_state)
            action_onehot = np.zeros([self.output_shape])
            action_onehot[action] = 1
            actions.append(action_onehot)
            rewards.append(reward)
            dones.append(done)
            state = next_state
            score += reward
            
            if done:

                self.lock.acquire()
                # reshape memory to appropriate shape for training
                states = np.vstack(states)
                actions = np.vstack(actions)
                next_states = np.vstack(next_states)
                dones = np.vstack(dones)
                discounted_r = np.vstack(self.discount_rewards(rewards))

                for e in range(self.epochs):

                    idx = np.random.randint(low=0, high=(states.shape[0]-self.mini_batch_size))
                    
                    action_batch = actions[idx:idx+self.mini_batch_size]
                    reward_batch = discounted_r[idx:idx+self.mini_batch_size]
                    done_batch= dones[idx:idx+self.mini_batch_size]
                    
                    # Critic part
                        
                    values = self.Critic(states[idx:idx+self.mini_batch_size])
                    values_next = self.Critic(next_states[idx:idx+self.mini_batch_size])
                    loss_critic = tf.reduce_mean(tf.math.square(values-reward_batch))

                    # Actor part 
                        
                    prob = self.Actor(states[idx:idx+self.mini_batch_size])
                    advantages = reward_batch - values + self.gamma *values_next*np.invert(done_batch).astype(np.float32)
                    advantages = tf.reshape(advantages, (-1))
                    log_prob = tf.math.log(tf.reduce_sum(tf.math.multiply(prob,action_batch),axis=1)+self.zero_fixer)
                    loss_actor = - tf.reduce_mean(log_prob*advantages)

                    entropy_coeff = 0.01
                    z0 = tf.reduce_sum(prob + self.zero_fixer, axis = 1)
                    z0 = tf.stack([z0,z0,z0,z0,z0,z0], axis=-1)
                    p0 = prob / z0 
                    entropy = tf.reduce_sum(p0 * (tf.math.log(p0 + self.zero_fixer)), axis=-1)
                    mean_entropy = tf.reduce_mean(entropy) 
                    entropy_loss =  mean_entropy * entropy_coeff 

                    self.loss_acc[e] += loss_actor + entropy_loss + 0.5 * loss_critic

                    
                print("episode: {}/{}, thread: {}, score: {}, average: {:.2f} {}".format(self.episode, self.episodes, thread, score, 0, 0))    
                self.episode += 1
                self.lock.release()
                #if self.step % 10==0:
                 #   print(values,prob)
        self.env.close()

    def update_weights(self, thread):

        self.states, self.actions, self.rewards, self.next_states, self.dones, self.predictions, self.values = self.call_memory(thread)

        index = np.arange(len(self.rewards))

        for e in range(self.epochs):
            
            np.random.shuffle(index)
            step_size = len(self.rewards)// self.mini_batch_size

            for start in range(0,len(self.rewards), step_size):
                
                end = start + step_size
                idx = index[start:end]

                states = self.states[idx]
                actions = self.actions[idx]
                rewards = self.rewards[idx]
                next_states = self.next_states[idx]
                dones = self.dones[idx]
                old_prob = self.predictions[idx]
                old_values = self.values[idx]
                    
                # Critic part
                cliprange = 0.2

                values = self.Critic(states)
                values_next = self.Critic(next_states)
                values_clipped = old_values + tf.clip_by_value(values - old_values, - cliprange, cliprange)
                values_loss_1 = tf.math.square(values - rewards)                  
                values_loss_2 = tf.math.square(values_clipped - rewards)
                                    
                self.loss_critic = tf.reduce_mean(tf.math.maximum(values_loss_1, values_loss_2))*0.5

                # Actor part 
                prob = self.Actor(states)

                advantages = rewards - values + self.gamma *values_next*np.invert(dones).astype(np.float32)
                advantages = np.reshape(advantages, (-1))
                
                log_prob = tf.reduce_sum(tf.math.multiply((prob+self.zero_fixer),actions),axis=1)
                log_old_prob = tf.reduce_sum(tf.math.multiply((old_prob+self.zero_fixer),actions),axis=1)

                clipping_value = 0.2
                r = log_prob/(log_old_prob+self.zero_fixer)
                r1 = - advantages * r
                r2 = - advantages * tf.clip_by_value(r, 1 - clipping_value, 1 + clipping_value)
                
                entropy_coeff = 0.01
                z0 = tf.reduce_sum(prob, axis = 1)
                z0 = tf.stack([z0,z0,z0,z0,z0,z0], axis=-1)
                p0 = prob / (z0 + self.zero_fixer) 
                entropy = tf.reduce_sum(p0 * (tf.math.log(p0 + self.zero_fixer)), axis=-1)
                mean_entropy = tf.reduce_mean(entropy) 
                self.entropy_loss =  mean_entropy * entropy_coeff 
                
                self.loss_actor = tf.math.reduce_mean(tf.math.maximum(r1,r2), axis=None) - self.entropy_loss

                self.loss_acc += self.loss_actor + self.loss_critic

            
    def play_game(self, thread):        
        
        frame = self.env.reset()
        state = self.GetImage(frame)
        
        done = False

        self.loss_actor = 0
        self.loss_critic = 0 
        self.entropy_loss = 0

        self.states, self.actions, self.rewards, self.next_states, self.dones, self.predictions, self.values = [], [], [], [], [], [], []

        while not done:

            #self.env.render()

            prediction = self.Actor(state)[0]
            value = self.Critic(state)[0]
            self.values.append(value)
            self.predictions.append(prediction)             
            action = np.random.choice(self.output_shape, p=prediction.numpy())
            next_state, reward, done, _ = self.env.step(action)
            next_state = self.GetImage(next_state)

            self.states.append(state)
            self.next_states.append(next_state)
            action_onehot = np.zeros([self.output_shape])
            action_onehot[action] = 1
            self.actions.append(action_onehot)
            self.rewards.append(reward)
            self.dones.append(done)
            state = next_state
            self.score += reward
            
            if done:

                # reshape memory to appropriate shape for training
                self.states = np.vstack(self.states)
                self.actions = np.vstack(self.actions)
                self.next_states = np.vstack(self.next_states)
                self.dones = np.vstack(self.dones)
                discounted_r = np.vstack(self.discount_rewards(self.rewards))
                self.predictions = np.vstack(self.predictions)
                self.values = np.vstack(self.values)

                self.save_to_memory(self.states, self.actions, discounted_r, self.next_states, self.dones, self.predictions, self.values, thread)

                #if self.step % 10==0:
                 #   print(values,prob)
        self.env.close()

    def save_to_memory(self, states, actions, rewards, next_states, dones, predictions, values, thread):

        self.memory[thread] = states, actions, rewards, next_states, dones, predictions, values


    def call_memory(self, thread):


        return self.memory[thread]

    def clear_memory(self):

        self.memory = [[] for _ in range(self.batch_size)]

    def normal_run(self):
        
        for step in range(self.episodes):
            
            frame = self.env.reset()
            state = self.GetImage(frame)
            done = False
            score = 0

            start = time.time()
            self.states, self.actions, self.rewards, self.predictions, self.values, self.advantages, self.dones, self.next_states = [],[], [], [], [], [], [],[]

            
            #print(self.name)
            while not done:

                self.env.render()
                prediction = self.Actor(state)[0]
                value = self.Critic(state)[0]
                self.predictions.append(prediction)       
                self.values.append(value)      

                action = np.random.choice(self.output_shape, p=prediction.numpy())
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.GetImage(next_state)

                self.states.append(state)
                self.next_states.append(next_state)
                action_onehot = np.zeros([self.output_shape])
                action_onehot[action] = 1
                self.actions.append(action_onehot)
                self.rewards.append(reward)
                self.dones.append(done)
                state = next_state
                score += reward
                end = time.time()

                if done:
                    

                    # reshape memory to appropriate shape for training
                    
                    self.states = np.vstack(self.states)
                    self.actions = np.vstack(self.actions)
                    self.predictions = np.vstack(self.predictions)
                    self.next_states = np.vstack(self.next_states)
                    self.dones = np.vstack(self.dones)
                    self.rewards = np.vstack(self.discount_rewards(self.rewards))
                    self.values = np.vstack(self.values)

                    index = np.arange(len(self.rewards))

                    for e in range(self.epochs):
                        '''
                        idx = np.random.randint(low=0, high=(self.states.shape[0]-self.mini_batch_size))
                        
                        states = self.states[idx:idx+self.mini_batch_size]
                        actions = self.actions[idx:idx+self.mini_batch_size]
                        rewards = self.rewards[idx:idx+self.mini_batch_size]
                        next_states = self.next_states[idx:idx+self.mini_batch_size]
                        dones = self.dones[idx:idx+self.mini_batch_size]
                        old_prob = self.predictions[idx:idx+self.mini_batch_size]
                        old_values = self.values[idx:idx+self.mini_batch_size]
                        '''
                        np.random.shuffle(index)
                        step_size = len(self.rewards)// self.mini_batch_size

                        for start in range(0,len(self.rewards), step_size):
                            
                            end = start + step_size
                            idx = index[start:end]

                            states = self.states[idx]
                            actions = self.actions[idx]
                            rewards = self.rewards[idx]
                            next_states = self.next_states[idx]
                            dones = self.dones[idx]
                            old_prob = self.predictions[idx]
                            old_values = self.values[idx]
                            

                            with tf.GradientTape() as tape:
                                
                                # Critic part
                                cliprange = 0.2
            
                                values = self.Critic(states)
                                values_next = self.Critic(next_states)
                                values_clipped = old_values + tf.clip_by_value(values - old_values, - cliprange, cliprange)
                                values_loss_1 = tf.math.square(values - rewards)                  
                                values_loss_2 = tf.math.square(values_clipped - rewards)
                                                    
                                self.loss_critic = tf.reduce_mean(tf.math.maximum(values_loss_1, values_loss_2))*0.5
                                grads = tape.gradient(self.loss_critic, self.Critic.trainable_weights)
                                self.optimizer_c.apply_gradients(zip(grads, self.Critic.trainable_weights))
                            
                            with tf.GradientTape() as tape: 

                                # Actor part 
                                prob = self.Actor(states)

                                advantages = rewards - values + self.gamma *values_next*np.invert(dones).astype(np.float32)
                                advantages = np.reshape(advantages, (-1))
                                
                                log_prob = tf.reduce_sum(tf.math.multiply((prob+self.zero_fixer),actions),axis=1)
                                log_old_prob = tf.reduce_sum(tf.math.multiply((old_prob+self.zero_fixer),actions),axis=1)

                                clipping_value = 0.2
                                r = log_prob/(log_old_prob+self.zero_fixer)
                                r1 = - advantages * r
                                r2 = - advantages * tf.clip_by_value(r, 1 - clipping_value, 1 + clipping_value)
                                
                                entropy_coeff = 0.01
                                z0 = tf.reduce_sum(prob, axis = 1)
                                z0 = tf.stack([z0,z0,z0,z0,z0,z0], axis=-1)
                                p0 = prob / (z0 + self.zero_fixer) 
                                entropy = tf.reduce_sum(p0 * (tf.math.log(p0 + self.zero_fixer)), axis=-1)
                                mean_entropy = tf.reduce_mean(entropy) 
                                self.entropy_loss =  mean_entropy * entropy_coeff 
                                
                                self.loss_actor = tf.math.reduce_mean(tf.math.maximum(r1,r2), axis=None) - self.entropy_loss

                                grads = tape.gradient(self.loss_actor, self.Actor.trainable_weights)
                                self.optimizer_a.apply_gradients(zip(grads, self.Actor.trainable_weights))                    
                                
                    if step % 1 == 0: #and self.setup == 'normal':
                        self.logger.log_performance(step, score, 0, self.loss_critic.numpy(), 0, 0, values[0:5], self.optimizer._decayed_lr(tf.float32).numpy(), prob[0:5])
                
                    if step % self.save_every == 0 and self.setup == 'normal':
                        self.Actor.save('{}/{}_actor.h5'.format(self.save_path, step))
                        self.Critic.save('{}/{}_critic.h5'.format(self.save_path, step))          
                    

        self.env.close()

        if self.setup == 'genetic':
            return loss_actor, loss_critic, entropy_loss, self.predictions[0:4], score
        

    def discount_rewards(self, reward):
        # Compute the gamma-discounted rewards over an episode
        gamma = 0.98   
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            if reward[i] != 0: # reset the sum, since this was a game boundary (pong specific!)
                running_add = 0
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        #discounted_r -= np.min(discounted_r) # normalizing the result
        #discounted_r = discounted_r*0.5
        #discounted_r /= tf.reduce_sum(discounted_r,axis=0) # divide by standard deviation
        return discounted_r

    
    def GetImage(self, frame):
        # croping frame to 80x80 size
        frame_cropped = frame[35:195:2, ::2,:]
        frame_rgb = 0.299*frame_cropped[:,:,0] + 0.587*frame_cropped[:,:,1] + 0.114*frame_cropped[:,:,2]

        # convert everything to black and white (agent will train faster)
        frame_rgb[frame_rgb < 100] = 0
        frame_rgb[frame_rgb >= 100] = 255    

        # dividing by 255 we expresses value to 0-1 representation
        new_frame = np.array(frame_rgb).astype(np.float32) / 255.0
        new_frame = new_frame[np.newaxis, np.newaxis,:,:]

        return new_frame 


