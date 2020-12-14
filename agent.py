# Tutorial by www.pylessons.com
# Tutorial written for - Tensorflow 1.15, Keras 2.2.4

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import random
import gym
import pylab
import numpy as np
import tensorflow as tf 
import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten,MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras import backend as K
import cv2
import new_nn

tf.keras.backend.clear_session()

class A2CAgent:
    # Actor-Critic Main Optimization Algorithm
    def __init__(self, env_name):
        # Initialization
        # Environment and PPO parameters
        self.env_name = env_name       
        self.env = gym.make(env_name)
        self.action_size = self.env.action_space.n

        print(self.action_size, 'tjhis asdfask fadskj aslködfjf sakösjflö ')
        self.EPISODES, self.max_average = 10000, -21.0 # specific for pong
        self.lr = 0.000025

        self.ROWS = 80
        self.COLS = 80
        self.REM_STEP = 4

        # Instantiate games and plot memory
        self.states, self.actions, self.rewards = [], [], []
        self.scores, self.episodes, self.average = [], [], []

        self.Save_Path = 'Models'
        self.state_size = (self.REM_STEP, self.ROWS, self.COLS)
        self.image_memory = np.zeros(self.state_size)
        
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.path = '{}_A2C_{}'.format(self.env_name, self.lr)
        self.Model_name = os.path.join(self.Save_Path, self.path)

        self.hypernetwork = new_nn.Hypernetwork_PONG('this is the motherfucking network')
        self.hypernetwork.compile(loss = 'categorical_crossentropy', optimizer=RMSprop(lr=0.0005))

            # Create Actor-Critic network model


        self.Actor = new_nn.Actor(input_shape=self.state_size, output_shape = self.action_size, lr=self.lr)
        #self.Actor.built = True
        #dont compile the actor network for the hypernetwork
        
        self.Actor.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=self.lr))
        print(self.Actor.summary())
        #self.Actor.compiled_loss(self.action_size, self.action_size, sample_weight=self.action_size)
        #print(self.Actor.summary())

        self.Critic = new_nn.Critic(input_shape=self.state_size, output_shape = self.action_size, lr=self.lr)
        self.Critic.compile(loss='mse', optimizer=RMSprop(lr=self.lr))

    @tf.function
    def remember(self, state, action, reward):
        # store episode actions to memory
        self.states.append(state)
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        self.actions.append(action_onehot)
        self.rewards.append(reward)

    @tf.function
    def act(self, state):
        # Use the network to predict the next action to take, using the model
        prediction = self.Actor(state)[0]
        print(prediction)
        return prediction

    #@tf.function
    def discount_rewards(self, reward):
        # Compute the gamma-discounted rewards over an episode
        gamma = 0.99    # discount rate
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

    def replay_hypernetwork(self):
        
        # reshape memory to appropriate shape for training
        states = np.vstack(self.states)
        actions = np.vstack(self.actions)

        # Compute discounted rewards
        discounted_r = self.discount_rewards(self.rewards)

        # Get Critic network predictions
        values = self.Critic.predict(states)[:, 0]
        # Compute advantages
        advantages = discounted_r - values
        # training Actor and Critic networks
        print('now the hypernetwork is updated')
        
        optimizer = RMSprop()
        loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
        loss = loss_fn(actions,advantages)
        grads = tape.gradients(loss, self.hypernetwork.trainable_weigths)
        optimizer.apply_gradients(zip(grads, self.hypernetwork.trainable_weights))


        #self.hypernetwork.fit(states, actions, sample_weight=advantages, epochs=1, verbose=0)
        print('now the hypernetwork has been updated')
        self.Critic.fit(states, discounted_r, epochs=1, verbose=0)
        # reset training memory
        self.states, self.actions, self.rewards = [], [], []


    def replay(self):
        # reshape memory to appropriate shape for training
        states = np.vstack(self.states)
        actions = np.vstack(self.actions)

        # Compute discounted rewards
        discounted_r = self.discount_rewards(self.rewards)

        # Get Critic network predictions
        values = self.Critic.predict(states)[:, 0]
        # Compute advantages
        advantages = discounted_r - values
        # training Actor and Critic networks
        self.Actor.fit(states, actions, sample_weight=advantages, epochs=1, verbose=0)
        self.Critic.fit(states, discounted_r, epochs=1, verbose=0)
        # reset training memory
        self.states, self.actions, self.rewards = [], [], []
    
    def load(self, Actor_name, Critic_name):
        self.Actor = load_model(Actor_name, compile=False)
        #self.Critic = load_model(Critic_name, compile=False)

    def save(self):
        self.Actor.save(self.Model_name + '_Actor.h5')
        #self.Critic.save(self.Model_name + '_Critic.h5')

    pylab.figure(figsize=(18, 9))
    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        if str(episode)[-2:] == "00":# much faster than episode % 100
            pylab.plot(self.episodes, self.scores, 'b')
            pylab.plot(self.episodes, self.average, 'r')
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
            try:
                pylab.savefig(self.path+".png")
            except OSError:
                pass

        return self.average[-1]

    def imshow(self, image, rem_step=0):
        cv2.imshow(self.Model_name+str(rem_step), image[rem_step,...])
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return

    def GetImage(self, frame):
        # croping frame to 80x80 size
        frame_cropped = frame[35:195:2, ::2,:]
        if frame_cropped.shape[0] != self.COLS or frame_cropped.shape[1] != self.ROWS:
            # OpenCV resize function 
            frame_cropped = cv2.resize(frame, (self.COLS, self.ROWS), interpolation=cv2.INTER_CUBIC)
        
        # converting to RGB (numpy way)
        frame_rgb = 0.299*frame_cropped[:,:,0] + 0.587*frame_cropped[:,:,1] + 0.114*frame_cropped[:,:,2]

        # convert everything to black and white (agent will train faster)
        frame_rgb[frame_rgb < 100] = 0
        frame_rgb[frame_rgb >= 100] = 255
        # converting to RGB (OpenCV way)
        #frame_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_RGB2GRAY)     

        # dividing by 255 we expresses value to 0-1 representation
        new_frame = np.array(frame_rgb).astype(np.float32) / 255.0

        # push our data by 1 frame, similar as deq() function work
        self.image_memory = np.roll(self.image_memory, 1, axis = 0)

        # inserting new frame to free space
        self.image_memory[0,:,:] = new_frame

        # show image frame   
        #self.imshow(self.image_memory,0)
        #self.imshow(self.image_memory,1)
        #self.imshow(self.image_memory,2)
        #self.imshow(self.image_memory,3)
        
        return np.expand_dims(self.image_memory, axis=0)

    def reset(self):
        frame = self.env.reset()
        for i in range(self.REM_STEP):
            state = self.GetImage(frame)
        return state

    #@tf.function
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state = self.GetImage(next_state)
        return next_state, reward, done, info
    
    def set_weights(self, weights_actor):
        
        k1 = 81936-16
        b1 = 81936
        k3 = b1 + 8224-32
        b3 = b1 + 8224
        k6 = b3 +82176 - 256
        b6 = b3 +82176
        k7 = b6 + 1542 - 6
        b7 = b6 + 1542

        self.Actor = Actor(input_shape=self.state_size, output_shape = self.action_size, lr=self.lr)
        self.Actor.built = True
        self.Actor.layers[2].kernel = np.reshape(weights_actor[:k1],(self.Actor.layers[2].kernel.shape))
        self.Actor.layers[4].kernel = np.reshape(weights_actor[b1:k3],(self.Actor.layers[4].kernel.shape))
        self.Actor.layers[7].kernel = np.reshape(weights_actor[b3:k6],(self.Actor.layers[7].kernel.shape))
        self.Actor.layers[8].kernel = np.reshape(weights_actor[b6:k7],(self.Actor.layers[8].kernel.shape))


    def run_hypernetwork(self):


        Actor = new_nn.Actor(self.state_size, self.action_size, lr=self.lr)

        
        for i in range(self.EPISODES):
            state = self.reset()
            done, score, SAVING = False, 0, ''


            with tf.GradientTape() as tape:
                z  = np.random.uniform(low = -1, high = 1, size = 1000)
                z = tf.constant(z[np.newaxis,:])

                weights_actor = self.hypernetwork(z)

                

                k1 = 81936-16
                b1 = 81936
                k3 = b1 + 8224-32
                b3 = b1 + 8224
                k6 = b3 +82176 - 256
                b6 = b3 +82176
                k7 = b6 + 1542 - 6
                b7 = b6 + 1542

                Actor.layers[1].kernel = tf.reshape(weights_actor[:k1],(Actor.layers[1].kernel.shape))
                Actor.layers[3].kernel = tf.reshape(weights_actor[b1:k3],(Actor.layers[3].kernel.shape))
                Actor.layers[6].kernel = tf.reshape(weights_actor[b3:k6],(Actor.layers[6].kernel.shape))
                Actor.layers[7].kernel = tf.reshape(weights_actor[b6:k7],(Actor.layers[7].kernel.shape))

                predictions = []

                while not done:
                    self.env.render()

                    prediction = Actor(state)[0]
                    predictions.append(prediction)
                    
                    action = np.random.choice(self.action_size, p=prediction.numpy())
                    #tape.watch(tf.convert_to_tensor(action))

                    next_state, reward, done, info = self.env.step(action)
                    next_state = self.GetImage(next_state)


                    #self.remember(state,action,reward)

                    self.states.append(state)
                    action_onehot = np.zeros([self.action_size])
                    action_onehot[action] = 1
                    self.actions.append(action_onehot)
                    self.rewards.append(reward)
                    state = next_state
                    score += reward
                    
                    if done:

                        # reshape memory to appropriate shape for training
                        self.states = np.vstack(self.states)
                        self.actions = np.vstack(self.actions)
                        predictions = tf.stack(predictions)
                        #self.actions = tf.convert_to_tensor(self.actions)

                        # Compute discounted rewards
                        discounted_r = self.discount_rewards(self.rewards)

                        # Get Critic network predictions
                        values = self.Critic(self.states)[:, 0]
                        # Compute advantages
                        self.advantages = discounted_r - values.numpy()
                    
                        optimizer = RMSprop(self.lr)
                        loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
                      
                        self.actions = self.actions * self.advantages[:,np.newaxis]
                        self.actions = self.actions * (-1)
                      
                        loss = loss_fn(predictions, self.actions)

                        grads = tape.gradient(loss, self.hypernetwork.trainable_weights)
                        optimizer.apply_gradients(zip(grads, self.hypernetwork.trainable_weights))


                        #self.hypernetwork.fit(states, actions, sample_weight=advantages, epochs=1, verbose=0)
                        self.Critic.fit(self.states, discounted_r, epochs=1, verbose=0)
                        # reset training memory
                        print("this is the motherfucking reward", score, 'episode', i)
                        self.states, self.actions, self.rewards = [], [], []

                        #print("episode: {}/{}, score: {}, average: {:.2f} {}".format(e, self.EPISODES, score, average, SAVING))
                        
                        
        self.env.close()


    def run(self):
        for e in range(self.EPISODES):
            state = self.reset()
            done, score, SAVING = False, 0, ''
            while not done:
                self.env.render()
                # Actor picks an action
                action = self.act(state)
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, _ = self.step(action)
                # Memorize (state, action, reward) for training
                self.remember(state, action, reward)
                # Update current state
                state = next_state
                score += reward
                if done:
                    average = self.PlotModel(score, e)
                    # saving best models
                    if average >= self.max_average:
                        self.max_average = average
                        self.save()
                        SAVING = "SAVING"
                    else:
                        SAVING = ""
                    print("episode: {}/{}, score: {}, average: {:.2f} {}".format(e, self.EPISODES, score, average, SAVING))

                    self.replay()
        # close environemnt when finish training
        self.env.close()

    def test(self, Actor_name, Critic_name):
        self.load(Actor_name, Critic_name)
        for e in range(100):
            state = self.reset()
            done = False
            score = 0
            while not done:
                action = np.argmax(self.Actor.predict(state))
                state, reward, done, _ = self.step(action)
                score += reward
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, score))
                    break
        self.env.close()

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

        for e in range(1):
            state = self.reset()
            state = np.array(state, dtype = 'float32')
            done = False
            score = 0
            while not done:
                #self.env.render()
                output = self.Actor(state)[0]
                action = np.argmax(output)
                state, reward, done, _ = self.step(action)
                score += reward
                if done:
                    average.append(score)
                    #print("episode: {}/{}, score: {}".format(e, self.EPISODES, score))
                    break
                
        self.env.close()

        return np.mean(average)


if __name__ == "__main__":
    #env_name = 'PongDeterministic-v4'
    #env_name = 'Pong-v0'
    #agent = A2CAgent(env_name)
    #agent.run_hypernetwork()
    #agent.test('Pong-v0_A2C_2.5e-05_Actor.h5', '')
    #agent.test('PongDeterministic-v4_A2C_1e-05_Actor.h5', '')
    pass