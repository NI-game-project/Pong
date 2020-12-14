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
import cv2
import networks

tf.keras.backend.clear_session()

class A2C_Agent:
    # Actor-Critic Main Optimization Algorithm
    def __init__(self, env_name, setup):
        
        # Initialization
        self.setup = setup
        self.env_name = env_name       
        self.env = gym.make(env_name)
        self.output_shape = self.env.action_space.n

        #hyperperameters
        self.episodes = 10000
        self.epochs = 5
        self.batch_size = 8
        self.lr = 0.0001

        #input shape of the pong enviroment
        self.rows = 80
        self.cols = 80
        self.channels = 1


        # Instantiate games and plot memory
        self.states, self.actions, self.rewards, self.predictions = [], [], [], []
        self.scores, self.average = [], []

        self.Save_Path = 'models/pong_hypernetwork'
        self.input_shape = (self.channels, self.rows, self.cols)
        self.image_memory = np.zeros(self.input_shape)
        
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.path = '{}_A2C_{}'.format(self.env_name, self.lr)
        self.Model_name = os.path.join(self.Save_Path, self.path)


        #Initializiation of the hypernetwork
        if self.setup == 'hypernetwork':

            self.hypernetwork = networks.Hypernetwork_PONG_first('Hypernetwork_Pong')
            self.hypernetwork.compile(optimizer=RMSprop(lr=self.lr))

        #TODO: combine the two networks into one and loading different ones for each configuration

        #the Actor network
        self.Actor = networks.Actor(input_shape=self.input_shape, output_shape = self.output_shape, lr=self.lr)  
        #self.Actor.compile(optimizer=RMSprop(lr=self.lr))

        #print the Actor network
        print(self.Actor.summary())

        #the Critic network
        if self.setup == 'hypernetwork' or self.setup == 'normal':

            self.Critic = networks.Critic(input_shape=self.input_shape, output_shape = 1, lr=self.lr)
            self.Critic.compile(loss='mse', optimizer=RMSprop(lr=self.lr))
    
    # this is the function, which takes the generated weigths by the hypernetwork and sets them as the kernels and biases of the Actor
    def set_weights(self, weights):

        last_used = 0

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

    # this is the training function for the hypernetwork
    def run_hypernetwork(self):
        
        #first initialize the optimizer and the embedding vector
        #TODO: different types of hypernetworks. create a version with a baysian input

        optimizer = keras.optimizers.Adam(lr=self.lr)
        z = np.random.uniform(low = -1, high = 1, size = 300)
        z = z[np.newaxis,:]
        #iterate the training procedure over the number of episodes
        for i in range(self.episodes):
            
            #resest the enviroment and set the score to zero
            frame = self.env.reset()
            state = self.GetImage(frame)
            done = False
            score = 0
            
            #while the enviroment doesnt return a done value
            while not done:
                
                #uncomment this to see the training
                self.env.render()
                
                #make prediciton with the first state
                prediction = self.Actor(state)[0]
                self.predictions.append(prediction)
                #select a action with the given probability and get the according next_step from the enviroment
                action = np.random.choice(self.output_shape, p=prediction.numpy())
                next_state, reward, done, _ = self.env.step(action)
                #transform the output of the the enviroment into the right data format
                next_state = self.GetImage(next_state)

                self.states.append(state)
                #create a onehot dataformat of the predictions as actions 
                action_onehot = np.zeros([self.output_shape])
                action_onehot[action] = 1
                self.actions.append(action_onehot)
                self.rewards.append(reward)
                state = next_state
                score += reward
                #if score < -4: 
                #    done = True
                #once the episode is finished do:
                if done:
                    
                    # reshape memory to appropriate shape for training
                    self.states = np.vstack(self.states)
                    self.actions = np.vstack(self.actions)
                    self.predictions = np.vstack(self.predictions)
                
                    
                    # Compute discounted rewards
                    discounted_r = np.vstack(self.discount_rewards(self.rewards))

                    # Get Critic network predictions
                    values = self.Critic(self.states, training=True)

                    # Compute advantages
                    self.advantages = discounted_r - values

                    y_true = np.hstack([self.advantages, self.predictions, self.actions])
                    
                    #iterate over the number of epochs
                    for e in range(self.epochs):

                        #TODO: this shuffling is pretty ugly, think of a better way
                        np.random.seed(e)
                        np.random.shuffle(y_true)
                        np.random.seed(e)
                        np.random.shuffle(self.states)
                        
                        #unpack the data again
                        advantages, predictions, actions = y_true[:, :1], y_true[:, 1:1+self.output_shape], y_true[:, 1+self.output_shape:]
                        
                        #transform them into tensorflow constants
                        advantages = tf.constant(advantages, dtype='float32')
                        predictions = tf.constant(predictions, dtype='float32')
                        actions = tf.constant(predictions, dtype='float32')

                        with tf.GradientTape() as tape:
                            
    
                            weights = self.hypernetwork(z)
                            #print(self.hypernetwork.summary())
                            
                            #set the weights
                            self.set_weights(weights)
                            #make predictions with the given weights
                            y_pred_actor = self.Actor(self.states)

                            #Do a proximal policy calculation of the loss function with the values of the critic
                            loss_clipping = tf.constant([0.2])
                            entropy_loss = tf.constant([5e-3])

                            prob = y_pred_actor * actions
                            old_prob = actions * predictions
                            r = prob/(old_prob + tf.constant([1e-10]))
                            p1 = r * advantages
                            p2 = K.clip(r, min_value=1-loss_clipping, max_value=1+loss_clipping) * advantages
                            loss = - K.mean(K.minimum(p1,p2) + entropy_loss * -(prob*K.log(prob + tf.constant([1e-10]))))

                            #calculate the gradients and update the weights of the the hypernetwork accordingly
                            grads = tape.gradient(loss, self.hypernetwork.trainable_weights)
                            optimizer.apply_gradients(zip(grads, self.hypernetwork.trainable_weights))

                            #self.Actor.fit(self.states, y_true, epochs=self.EPOCHS, verbose=0, shuffle=True, batch_size=len(self.rewards))
                            self.Critic.fit(self.states, discounted_r, epochs=1, verbose=0, shuffle=True, batch_size=len(self.rewards))



                    # reset training memory
                    print("this is the reward", score, 'episode', i)
                    self.states, self.actions, self.rewards, self.predictions, self.advantages = [], [], [], [], []

                    #print("episode: {}/{}, score: {}, average: {:.2f} {}".format(e, self.EPISODES, score, average, SAVING))
                        
                        
        self.env.close()

    def normal_run(self):

        optimizer = RMSprop(lr=self.lr)
        
        for i in range(self.episodes):
            
            frame = self.env.reset()
            state = self.GetImage(frame)
            
            done = False
            score = 0

            while not done:
                self.env.render()

                prediction = self.Actor(state, training=True)[0]
                self.predictions.append(prediction)
                
                action = np.random.choice(self.output_shape, p=prediction.numpy())
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.GetImage(next_state)

                self.states.append(state)
                action_onehot = np.zeros([self.output_shape])
                action_onehot[action] = 1
                self.actions.append(action_onehot)
                self.rewards.append(reward)
                state = next_state
                score += reward
                
                if done:

                    # reshape memory to appropriate shape for training
                    self.states = np.vstack(self.states)
                    self.actions = np.vstack(self.actions)
                    self.predictions = np.vstack(self.predictions)
                    #self.actions = tf.convert_to_tensor(self.actions)

                    # Compute discounted rewards
                    discounted_r = np.vstack(self.discount_rewards(self.rewards))

                    # Get Critic network predictions
                    values = self.Critic(self.states, training=True)

                    # Compute advantages
                    self.advantages = discounted_r - values

                    y_true = np.hstack([self.advantages, self.predictions, self.actions])


                    for e in range(self.epochs):

                        with tf.GradientTape() as tape:
                            np.random.seed(e)
                            np.random.shuffle(y_true)
                            np.random.seed(e)
                            np.random.shuffle(self.states)
            
                            advantages, predictions, actions = y_true[:, :1], y_true[:, 1:1+self.output_shape], y_true[:, 1+self.output_shape:]

                            y_pred_actor = self.Actor(self.states, training=True)
                            loss_clipping = 0.2
                            entropy_loss = 5e-3

                            prob = y_pred_actor * actions
                            old_prob = actions * predictions
                            r = prob/(old_prob + 1e-10)
                            p1 = r * advantages
                            p2 = K.clip(r, min_value=1-loss_clipping, max_value=1+loss_clipping) * advantages
                            loss = - K.mean(K.minimum(p1,p2) + entropy_loss * -(prob*K.log(prob + 1e-10)))

                            
                            grads = tape.gradient(loss, self.Actor.trainable_weights)
                            optimizer.apply_gradients(zip(grads, self.Actor.trainable_weights))

                            #self.Actor.fit(self.states, y_true, epochs=self.EPOCHS, verbose=0, shuffle=True, batch_size=len(self.rewards))
                            self.Critic.fit(self.states, discounted_r, epochs=1, verbose=0, shuffle=True, batch_size=len(self.rewards))



                    # reset training memory
                    print("this is the reward", score, 'episode', i)
                    self.states, self.actions, self.rewards, self.predictions, self.advantages = [], [], [], [], []

                    #print("episode: {}/{}, score: {}, average: {:.2f} {}".format(e, self.EPISODES, score, average, SAVING))
                        
                        
        self.env.close()


    def discount_rewards(self, reward):
        # Compute the gamma-discounted rewards over an episode
        gamma = 0.993    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            if reward[i] != 0: # reset the sum, since this was a game boundary (pong specific!)
                running_add = 0
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= np.std(discounted_r) # divide by standard deviation
        return discounted_r

    
    def load(self, Actor_name, Critic_name):
        self.Actor = load_model(Actor_name, compile=False)
        #self.Critic = load_model(Critic_name, compile=False)

    def save(self):
        self.Actor.save(self.Model_name + '_Actor.h5')
        #self.Critic.save(self.Model_name + '_Critic.h5')


    def GetImage(self, frame):
        # croping frame to 80x80 size
        frame_cropped = frame[35:195:2, ::2,:]
        if frame_cropped.shape[0] != self.cols or frame_cropped.shape[1] != self.rows:
            # OpenCV resize function 
            frame_cropped = cv2.resize(frame, (self.cols, self.rows), interpolation=cv2.INTER_CUBIC)
        
        # converting to RGB (numpy way)
        frame_rgb = 0.299*frame_cropped[:,:,0] + 0.587*frame_cropped[:,:,1] + 0.114*frame_cropped[:,:,2]

        # convert everything to black and white (agent will train faster)
        frame_rgb[frame_rgb < 100] = 0
        frame_rgb[frame_rgb >= 100] = 255    

        # dividing by 255 we expresses value to 0-1 representation
        new_frame = np.array(frame_rgb).astype(np.float32) / 255.0
        new_frame = new_frame[np.newaxis, np.newaxis,:,:]

        return new_frame 


    def step(self, action):

        next_state, reward, done, _ = self.env.step(action)
        next_state = self.GetImage(next_state)
        return next_state, reward, done
    
    def evaluate(self, weights):

        weights = np.array(weights, dtype = 'float32')
        last_used = 0
        average = []
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

        #change this according to the number of verification runs
        for e in range(1):
            frame = self.env.reset()
            state = self.GetImage(frame)
            state = np.array(state, dtype = 'float32')
            done = False
            score = 0
            while not done:
                #self.env.render()
                output = self.Actor(state)[0]
                action = np.argmax(output)
                state, reward, done = self.step(action)
                score += reward
                if done:
                    average.append(score)
                    #print("episode: {}/{}, score: {}".format(e, self.EPISODES, score))
                    break
                
        self.env.close()

        return np.mean(average)

#TODO: do several threads


if __name__ == "__main__":
    #env_name = 'PongDeterministic-v4'
    env_name = 'PongDeterministic-v4'
    setup = 'hypernetwork'
    agent = A2C_Agent(env_name, setup)
    agent.run_hypernetwork()
    #agent.run_normal()
    #agent.test('Pong-v0_A2C_2.5e-05_Actor.h5', '')
    #agent.test('PongDeterministic-v4_A2C_1e-05_Actor.h5', '')