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

tf.keras.backend.clear_session()

class A2C_Agent:
    # Actor-Critic Main Optimization Algorithm
    def __init__(self, env_name, save_path, setup, lr, batch_size, lamBda, episodes, decay_rate, save_every):
        
        # Initialization
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
        
        self.mini_batch_size = 64
        self.epochs = 8
        self.gamma = 0.95
        self.seed = 0
        self.zero_fixer = 1e-8

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

        
        self.save_path = save_path
        if not os.path.exists(self.save_path): os.makedirs(self.save_path)
        self.path = '{}_A2C_{}'.format(self.env_name, self.lr)
        self.Model_name = os.path.join(self.save_path, self.path)

        self.logger = logger.Logger(self.save_path)

        self.hypernetwork = networks.Hypernetwork_PONG('Hypernetwork_Pong')
        self.Actor = networks.Actor(input_shape=self.input_shape, output_shape = self.output_shape, seed=self.seed)  
        self.Critic = networks.Critic(input_shape=self.input_shape, output_shape = 1, seed=self.seed)
        
        self.lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = self.lr, decay_steps=1, decay_rate=self.decay_rate)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

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
    

    def run_hypernetwork(self):

        for step in range(self.episodes):
            
            with tf.GradientTape() as tape: 
                self.step = step

                self.loss_acc = 0

                z = np.random.uniform(low = -1, high = 1, size = [self.batch_size,1000])

                weights_1, weights_2, weights_3, weights_4, weights_5, weights_6 = self.hypernetwork(z, self.batch_size)

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

                self.w1 = w1_not_gauged
                self.w2 = w2_not_gauged
                self.w3 = w3_not_gauged
                self.w4 = w4_not_gauged
                self.w5 = w5_not_gauged
                self.w6 = w6_not_gauged

                self.b1 = b1_not_gauged
                self.b2 = b2_not_gauged
                self.b3 = b3_not_gauged
                self.b4 = b4_not_gauged
                self.b5 = b5_not_gauged
                self.b6 = b6_not_gauged
                
                weights_actor = tf.concat(axis=1,values=[tf.reshape(self.w1[:,:32,:], (self.batch_size,-1)), tf.reshape(self.b1[:,:32], (self.batch_size,-1)),\
                    tf.reshape(self.w2[:,:16,:], (self.batch_size,-1)), tf.reshape(self.b2[:,:16], (self.batch_size,-1)),\
                    tf.reshape(self.w3[:,:256,:], (self.batch_size,-1)), tf.reshape(self.b3[:,:256],(self.batch_size,-1)), \
                    tf.reshape(self.w4[:,:256,:], (self.batch_size,-1)), tf.reshape(self.b4[:,:256],(self.batch_size,-1)), \
                    tf.reshape(self.w5,(self.batch_size,-1)), tf.reshape(self.b5, (self.batch_size,-1))])
                
                weights_critic = tf.concat(axis=1, values=[tf.reshape(self.w1[:,32:,:],(self.batch_size,-1)), tf.reshape(self.b1[:,32:], (self.batch_size,-1)),\
                 tf.reshape(self.w2[:,16:,:],(self.batch_size,-1)), tf.reshape(self.b2[:,16:],(self.batch_size,-1)),
                  tf.reshape(self.w3[:,256:,:], (self.batch_size,-1)), tf.reshape(self.b3[:,256:],(self.batch_size,-1)), \
                    tf.reshape(self.w4[:,256:,:], (self.batch_size,-1)), tf.reshape(self.b4[:,256:],(self.batch_size,-1)), \
                        tf.reshape(self.w6,(self.batch_size,-1)), tf.reshape(self.b6, (self.batch_size,-1))])
                
                for num in range(self.batch_size):
                    self.score = 0
                    self.set_weights(weights_actor, weights_critic, num)
                    self.play_game()
                
                if self.batch_size > 1:

                    zero_fixer = 1e-8
                    input_noise_size = 300
                    noise_batch_size = tf.identity(self.batch_size,name='noise_batch_size') 

                    flattened_network = tf.concat(axis=1,values=[\
                            tf.reshape(self.w1, [noise_batch_size, -1]),tf.reshape(self.b1, [noise_batch_size, -1]),\
                            tf.reshape(self.w2, [noise_batch_size, -1]),tf.reshape(self.b2, [noise_batch_size, -1]),\
                            tf.reshape(self.w3, [noise_batch_size, -1]),tf.reshape(self.b3, [noise_batch_size, -1]),\
                            tf.reshape(self.w4, [noise_batch_size, -1]),tf.reshape(self.b4, [noise_batch_size, -1]),\
                            tf.reshape(self.w5, [noise_batch_size, -1]),tf.reshape(self.b5, [noise_batch_size, -1]),\
                            tf.reshape(self.w6, [noise_batch_size, -1]),tf.reshape(self.b6, [noise_batch_size, -1])])

                    # entropy estimated using  Kozachenko-Leonenko estimator, with l1 distances
                    mutual_distances = tf.math.reduce_sum(tf.math.abs(tf.expand_dims(flattened_network, 0) - tf.expand_dims(flattened_network, 1)), 2,name='mutual_squared_distances') # all distances between weight vector samples
                    nearest_distances = tf.identity(-1*tf.math.top_k(-1 * mutual_distances, k=2)[0][:, 1] ,name='nearest_distances') # distance to nearest neighboor for each weight vector sample
                    entropy_estimate = tf.identity(input_noise_size * tf.math.reduce_mean(tf.math.log(nearest_distances + zero_fixer)) + tf.math.digamma(tf.cast(noise_batch_size, tf.float32)), name='entropy_estimate')
                    loss_div = tf.identity( - 1 * entropy_estimate)
                    loss = self.loss_acc + self.lamBda * loss_div

                else:
                    loss_div = 0
                    loss = self.loss_acc *self.lamBda
        
                grads = tape.gradient(loss, self.hypernetwork.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.hypernetwork.trainable_weights))

                if step % 5 == 0:
                    
                    self.logger.log_performance(step, tf.reduce_sum(self.score).numpy(), self.loss_actor.numpy(), self.loss_critic.numpy(), self.entropy_loss.numpy(), loss_div.numpy(), loss.numpy(), self.optimizer._decayed_lr(tf.float32).numpy(), self.predictions[0:4] )
                
                if step % self.save_every == 0:
                    self.hypernetwork.save_weights('{}/{}_hypernetwork.h5'.format(self.save_path, step))

                if self.optimizer._decayed_lr(tf.float32).numpy() < 1e-5:
                    self.optimizer = Adam(lr=1e-5)

                self.predictions = []       

            
    def play_game(self):        
        
        frame = self.env.reset()
        state = self.GetImage(frame)
        
        done = False

        self.loss_actor = 0
        self.loss_critic = 0 
        self.entropy_loss = 0

        while not done:

            #self.env.render()

            prediction = self.Actor(state)[0]
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

                for e in range(self.epochs):

                    idx = np.random.randint(low=0, high=(self.states.shape[0]-self.mini_batch_size))
                    
                    actions = self.actions[idx:idx+self.mini_batch_size]
                    rewards = discounted_r[idx:idx+self.mini_batch_size]
                    dones = self.dones[idx:idx+self.mini_batch_size]
                    
                    # Critic part
                        
                    values = self.Critic(self.states[idx:idx+self.mini_batch_size])
                    values_next = self.Critic(self.next_states[idx:idx+self.mini_batch_size])
                    self.loss_critic = tf.reduce_mean(tf.math.square(values-rewards))

                    # Actor part 
                        
                    prob = self.Actor(self.states[idx:idx+self.mini_batch_size])
                    advantages = rewards - values + self.gamma *values_next*np.invert(dones).astype(np.float32)
                    advantages = tf.reshape(advantages, (-1))
                    log_prob = tf.math.log(tf.reduce_sum(tf.math.multiply(prob,actions),axis=1)+self.zero_fixer)
                    self.loss_actor = - tf.reduce_mean(log_prob*advantages)

                    entropy_coeff = 0.05
                    z0 = tf.reduce_sum(prob + self.zero_fixer, axis = 1)
                    z0 = tf.stack([z0,z0,z0,z0,z0,z0], axis=-1)
                    p0 = prob / z0 
                    entropy = tf.reduce_sum(p0 * (tf.math.log(p0 + self.zero_fixer)), axis=-1)
                    mean_entropy = tf.reduce_mean(entropy) 
                    self.entropy_loss =  mean_entropy * entropy_coeff 

                    self.loss_acc += self.loss_actor + self.entropy_loss + 0.5 * self.loss_critic
                    
                self.states, self.actions, self.rewards, self.predictions, self.advantages, self.dones, self.next_states = [], [], [], [], [], [],[]
                #if self.step % 10==0:
                 #   print(values,prob)
        self.env.close()
            


    def normal_run(self):
        
        for step in range(self.episodes):
            
            frame = self.env.reset()
            state = self.GetImage(frame)
            done = False
            score = 0

            while not done:

                #self.env.render()

                prediction = self.Actor(state)[0]
                
                self.predictions.append(prediction)
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
                score += reward
                
                if done:

                    # reshape memory to appropriate shape for training
                    self.states = np.vstack(self.states)
                    self.actions = np.vstack(self.actions)
                    self.predictions = np.vstack(self.predictions)
                    self.next_states = np.vstack(self.next_states)
                    self.dones = np.vstack(self.dones)
                    discounted_r = np.vstack(self.discount_rewards(self.rewards))

                    for e in range(self.epochs):

                        idx = np.random.randint(low=0, high=(self.states.shape[0]-self.mini_batch_size))
                        
                        actions = self.actions[idx:idx+self.mini_batch_size]
                        rewards = discounted_r[idx:idx+self.mini_batch_size]
                        dones = self.dones[idx:idx+self.mini_batch_size]
                        
                        with tf.GradientTape() as tape:
                            
                            values = self.Critic(self.states[idx:idx+self.mini_batch_size])
                            values_next = self.Critic(self.next_states[idx:idx+self.mini_batch_size])
                            loss_critic = tf.reduce_mean(tf.math.square(values-rewards))

                            grads = tape.gradient(loss_critic, self.Critic.trainable_weights)
                            self.optimizer.apply_gradients(zip(grads, self.Critic.trainable_weights))
                        
                        with tf.GradientTape() as tape: 
                            
                            prob = self.Actor(self.states[idx:idx+self.mini_batch_size])
                            advantages = rewards - values + self.gamma *values_next*np.invert(dones).astype(np.float32)
                            advantages = tf.reshape(advantages, (-1))
                            log_prob = tf.math.log(tf.reduce_sum(tf.math.multiply(prob,actions),axis=1)+1e-9)
                            loss_actor = - tf.reduce_mean(log_prob*advantages)

                            entropy_coeff = 0.1
                            a0 = prob - tf.reduce_max(prob, axis=-1, keepdims=True)
                            ea0 = tf.exp(a0)
                            z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
                            p0 = ea0 / z0
                            entropy = tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)
                            mean_entropy = tf.reduce_mean(entropy) 
                            entropy_loss = - mean_entropy * entropy_coeff 

                            loss = loss_actor + entropy_loss
                            grads = tape.gradient(loss, self.Actor.trainable_weights)
                            self.optimizer.apply_gradients(zip(grads, self.Actor.trainable_weights))


                    if step % 10 == 0 and self.setup == 'normal':
                        self.logger.log_performance(step, score, loss_actor.numpy(), loss_critic.numpy(), entropy_loss.numpy(), 0, loss.numpy(), self.optimizer._decayed_lr(tf.float32).numpy(), self.predictions[0:4])
                
                    if step % self.save_every == 0 and self.setup == 'normal':
                        self.Actor.save('{}/{}_actor.h5'.format(self.save_path, step))
                        self.Critic.save('{}/{}_critic.h5'.format(self.save_path, step))          
                    
                    self.states, self.actions, self.rewards, self.predictions, self.advantages, self.dones, self.next_states = [], [], [], [], [], [],[]

        self.env.close()

        if self.setup == 'genetic':
            return loss_actor, loss_critic, entropy_loss, self.predictions[0:4], score
        

    def discount_rewards(self, reward):
        # Compute the gamma-discounted rewards over an episode
        gamma = 0.99   
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            if reward[i] != 0: # reset the sum, since this was a game boundary (pong specific!)
                running_add = 0
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r) # normalizing the result
        #discounted_r /= np.std(discounted_r) # divide by standard deviation
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



