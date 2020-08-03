#Copyright (c) Microsoft Corporation. All rights reserved. 
#Licensed under the MIT License.


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
import numpy as np

def LFW_Generator(randomDim = 100, optim = Adam(lr=0.0002, beta_1=0.5)):
    """Creates a generateof for LFW dataset

    Args:
        randomDim (int, optional): input shape. Defaults to 100.
        optim ([Adam], optional): optimizer. Defaults to Adam(lr=0.0002, beta_1=0.5).
    """
    generator = Sequential()
    generator.add(Dense(512, input_dim=randomDim, kernel_initializer=initializers.RandomNormal(stddev=0.02),
                 name = 'layer'+str(np.random.randint(0,1e9))))
    generator.add(LeakyReLU(0.2,
                 name = 'layer'+str(np.random.randint(0,1e9))))
    generator.add(Dense(512,
                 name = 'layer'+str(np.random.randint(0,1e9))))
    generator.add(LeakyReLU(0.2,
                 name = 'layer'+str(np.random.randint(0,1e9))))
    generator.add(Dense(1024,
                 name = 'layer'+str(np.random.randint(0,1e9))))
    generator.add(LeakyReLU(0.2,
                 name = 'layer'+str(np.random.randint(0,1e9))))
    generator.add(Dense(2914, activation='tanh',
                 name = 'layer'+str(np.random.randint(0,1e9))))
    generator.compile(loss='binary_crossentropy', optimizer=optim)
    
    return generator


def LFW_Discriminator(optim = Adam(lr=0.0002, beta_1=0.5)):
    """Discriminator for LFW dataset

    Args:
        optim ([Adam], optional): optimizer. Defaults to Adam(lr=0.0002, beta_1=0.5).
    """
    discriminator = Sequential()
    discriminator.add(Dense(2048, input_dim=2914, kernel_initializer=initializers.RandomNormal(stddev=0.02),
                     name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(LeakyReLU(0.2,
                     name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(Dense(512,
                     name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(LeakyReLU(0.2,
                     name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(Dense(256,
                     name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(LeakyReLU(0.2,
                     name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(Dense(1, activation='sigmoid',
                     name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.compile(loss='binary_crossentropy', optimizer=optim)
    
    return discriminator

def LFW_DiscriminatorPrivate(OutSize = 2, optim = Adam(lr=0.0002, beta_1=0.5)):
    """The discriminator designed to guess which Generator generated the data

    Args:
        OutSize (int, optional): [description]. Defaults to 2.
        optim ([type], optional): optimizer. Defaults to Adam(lr=0.0002, beta_1=0.5).
    """
    discriminator = Sequential()
    discriminator.add(Dense(2048, input_dim=2914, kernel_initializer=initializers.RandomNormal(stddev=0.02),
                     name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(LeakyReLU(0.2,
                     name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(Dense(512,
                     name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(LeakyReLU(0.2,
                     name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(Dense(256,
                     name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(LeakyReLU(0.2,
                     name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.add(Dense(OutSize, activation='softmax',
                     name = 'layer'+str(np.random.randint(0,1e9))))
    discriminator.compile(loss='sparse_categorical_crossentropy', optimizer=optim)
    
    return discriminator