#Copyright (c) Microsoft Corporation. All rights reserved. 
#Licensed under the MIT License.


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LeakyReLU, Reshape, Conv2DTranspose, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
import numpy as np

def CIFAR_Generator(randomDim = 100, optim = Adam(lr=0.0002, beta_1=0.5)):
    """Creates a generateof for LFW dataset

    Args:
        randomDim (int, optional): input shape. Defaults to 100.
        optim ([Adam], optional): optimizer. Defaults to Adam(lr=0.0002, beta_1=0.5).
    """

    generator = Sequential()
    generator.add(Dense(2*2*512, input_shape=(randomDim,), kernel_initializer=initializers.RandomNormal(stddev=0.02),
                 name = 'layer'+str(np.random.randint(0,1e9))))
    generator.add(Reshape((2, 2, 512),
                 name = 'layer'+str(np.random.randint(0,1e9))))
    generator.add(LeakyReLU(0.2,
                 name = 'layer'+str(np.random.randint(0,1e9))))
    generator.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding='same',
                 name = 'layer'+str(np.random.randint(0,1e9))))
    generator.add(LeakyReLU(0.2,
                 name = 'layer'+str(np.random.randint(0,1e9))))
    generator.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same',
                 name = 'layer'+str(np.random.randint(0,1e9))))
    generator.add(LeakyReLU(0.2,
                 name = 'layer'+str(np.random.randint(0,1e9))))
    generator.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding='same',
                 name = 'layer'+str(np.random.randint(0,1e9))))
    generator.add(LeakyReLU(0.2,
                 name = 'layer'+str(np.random.randint(0,1e9))))
    generator.add(Conv2DTranspose(3, kernel_size=5, strides=2, padding='same',
                              activation='tanh',
                 name = 'layer'+str(np.random.randint(0,1e9))))
    generator.compile(loss='binary_crossentropy', optimizer=optim)
    
    return generator

def CIFAR_Discriminator(optim = Adam(lr=0.0002, beta_1=0.5)):
    """Discriminator for LFW dataset

    Args:
        optim ([Adam], optional): optimizer. Defaults to Adam(lr=0.0002, beta_1=0.5).
    """
    
    discriminator = Sequential()
    discriminator.add(Conv2D(64, kernel_size=5, strides=2, padding='same',
                             input_shape=((32, 32, 3)), kernel_initializer=initializers.RandomNormal(stddev=0.02),
                             name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(LeakyReLU(0.2,
                             name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(Conv2D(128, kernel_size=5, strides=2, padding='same',
                             name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(LeakyReLU(0.2,
                             name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(Conv2D(128, kernel_size=5, strides=2, padding='same',
                             name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(LeakyReLU(0.2,
                             name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(Conv2D(256, kernel_size=5, strides=2, padding='same',
                             name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(LeakyReLU(0.2,
                             name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(Flatten(name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(Dense(1, activation='sigmoid',
                             name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.compile(loss='binary_crossentropy', optimizer=optim)

    return discriminator

def CIFAR_DiscriminatorPrivate(OutSize = 2, optim = Adam(lr=0.0002, beta_1=0.5)):
    """The discriminator designed to guess which Generator generated the data

    Args:
        OutSize (int, optional): [description]. Defaults to 2.
        optim ([type], optional): optimizer. Defaults to Adam(lr=0.0002, beta_1=0.5).
    """
    
    discriminator = Sequential()
    discriminator.add(Conv2D(64, kernel_size=5, strides=2, padding='same',
                             input_shape=((32, 32, 3)), kernel_initializer=initializers.RandomNormal(stddev=0.02),
                             name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(LeakyReLU(0.2,
                             name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(Conv2D(128, kernel_size=5, strides=2, padding='same',
                             name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(LeakyReLU(0.2,
                             name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(Conv2D(128, kernel_size=5, strides=2, padding='same',
                             name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(LeakyReLU(0.2,name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(Conv2D(256, kernel_size=5, strides=2, padding='same',
                             name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(LeakyReLU(0.2,
                             name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(Flatten(name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(Dense(OutSize, activation='softmax',
                             name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.compile(loss='sparse_categorical_crossentropy', optimizer=optim)
    
    return discriminator


