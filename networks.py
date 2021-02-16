import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Input, Flatten, LeakyReLU
from tensorflow.keras.optimizers import RMSprop
from keras import Model


def Actor(input_shape, output_shape, seed):
    
    initializer = tf.keras.initializers.glorot_uniform(seed=seed)

    model = tf.keras.models.Sequential([
        Conv2D(32, (5, 5), padding="same", activation="relu", strides=(1,1),input_shape = input_shape, kernel_initializer=initializer),
        MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="same"),
        Conv2D(16, (5, 5), padding="same", strides=(1,1), activation="relu", kernel_initializer=initializer),
        MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="same"),
        tf.keras.layers.Flatten(),
        Dense(256, activation= "relu",kernel_initializer=initializer),
        Dense(256, activation= "relu",kernel_initializer=initializer),
        Dense(output_shape, activation= "softmax",kernel_initializer=initializer)
    ])
    
    return model

def Critic(input_shape, output_shape, seed):
        
        
    initializer = tf.keras.initializers.glorot_uniform(seed=seed)

    model = tf.keras.models.Sequential([
        Conv2D(32, (5, 5), padding="same", activation="relu", strides=(1,1),input_shape = input_shape, kernel_initializer=initializer),
        MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="same"),
        Conv2D(16, (5, 5), padding="same", strides=(1,1), activation="relu", kernel_initializer=initializer),
        MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="same"),
        tf.keras.layers.Flatten(),
        Dense(256, activation= "relu",kernel_initializer=initializer),
        Dense(256, activation= "relu",kernel_initializer=initializer),
        Dense(output_shape, activation= "linear",kernel_initializer=initializer)
    ])
    
    return model





class Hypernetwork_PONG(keras.Model):

    def __init__(self, name):
        super().__init__()
        kernel_init = tf.keras.initializers.VarianceScaling(scale=0.01, seed=42)
        bias_init = tf.keras.initializers.Constant(0)

        self.dense_1 = Dense(300, activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.dense_2 = Dense(2158*5, activation="relu",kernel_initializer=kernel_init, bias_initializer=bias_init)

        self.w1_dense_1 = Dense(100, activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.w1_dense_2 = Dense(2001, activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        
        self.w2_dense_1 = Dense(100, activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.w2_dense_2 = Dense(801, activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)

        self.w3_dense_1 = Dense(100, activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.w3_dense_2 = Dense(321, activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        
        self.w4_dense_1 = Dense(100, activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.w4_dense_2 = Dense(257, activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)

        self.w5_dense_1 = Dense(50, activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.w5_dense_2 = Dense(257, activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)

        self.w6_dense_1 = Dense(50, activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.w6_dense_2 = Dense(257, activation="relu", kernel_initializer=kernel_init, bias_initializer=bias_init)


    def call(self, inputs, batch_size):

        layer_1 = 32
        layer_2 = 16
        layer_3 = 256
        layer_4 = 256
        layer_5 = 6
        layer_6 = 1
        
        encoding = 5

        index_1 = layer_1*encoding*2
        index_2 = index_1 + layer_2*encoding*2
        index_3 = index_2 + layer_3*encoding*2
        index_4 = index_3 + layer_4*encoding*2
        index_5 = index_4 + layer_5*encoding
        index_6 = index_5 + layer_6*encoding    
        
        x = self.dense_1(inputs)
        x = self.dense_2(x)

        input_w1 = x[:,:index_1]
        input_w1 = tf.reshape(input_w1,(batch_size,layer_1*2,-1))
        w1 = self.w1_dense_1(input_w1)
        w1 = self.w1_dense_2(w1)

        input_w2 = x[:,index_1:index_2]
        input_w2 = tf.reshape(input_w2,(batch_size,layer_2*2,-1))
        w2 = self.w2_dense_1(input_w2)
        w2 = self.w2_dense_2(w2)

        input_w3 = x[:,index_2:index_3]
        input_w3 = tf.reshape(input_w3,(batch_size,layer_3*2,-1))
        w3 = self.w3_dense_1(input_w3)
        w3 = self.w3_dense_2(w3)
       
        input_w4 = x[:,index_3:index_4]
        input_w4 = tf.reshape(input_w4,(batch_size,layer_4*2,-1))
        w4 = self.w4_dense_1(input_w4)
        w4 = self.w4_dense_2(w4)
         

        input_w5 = x[:,index_4:index_5]
        input_w5 = tf.reshape(input_w5,(batch_size,layer_5,-1))
        w5 = self.w5_dense_1(input_w5)
        w5 = self.w5_dense_2(w5)       

        input_w6 = x[:,index_5:index_6]
        input_w6 = tf.reshape(input_w6,(batch_size,layer_6,-1))
        w6 = self.w6_dense_1(input_w6)
        w6 = self.w6_dense_2(w6)
        
        return w1, w2, w3, w4, w5, w6




