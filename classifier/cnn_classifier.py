# Copyright (c) Microsoft Corporation. All rights reserved. 
# Licensed under the MIT License.

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Reshape, Dense, Dropout, Flatten, LeakyReLU, Conv2D, MaxPooling2D

class CNNClassifier:
    """CNN classifier to classify images produced by different GAN models
    The classifier uses @Adadelta as optimizer
    """
    def __init__(self, num_classes, input_shape, dropout=0.5, learning_rate=1.0, rho=0.95, epsilon=1e-06):
        """Initializes and compiles the CNN classifier

        Args:
            num_classes (int): number of classes to be used for classification
            input_shape (tensor):  4D tensor shape of the input
            dropout (float, optional): dropout layer param. Defaults to 0.5.
            learning_rate (float, optional): learning rate of the optimizer. Defaults to 1.0.
            rho (float, optional): decay rate. Defaults to 0.95.
            epsilon (float optional): constant epsilon used to better conditioning the grad update. Defaults to 1e-06.
        """
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = self.__build_model()

    def train(self, x_train, y_train, x_validation, y_validation, batch_size=256, epochs=25):
        """Trains and evaluates the CNN classifier model.
        Uses accuracy as the metric

        Args:
            x_train (tensor): training data
            y_train (tensor): labels of the training data
            x_validation (tensor): validation batch input 
            y_validation (tensor): tensor representing labels of the validation batch
            batch_size (int): size of the batch per epoch. Defaults to 256
            epochs (int): number of epochs. Defaults to 25
        
        Returns:
        Scalar test loss - loss and accuracy post evaluation
        """
        self.model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_validation, y_validation))
        return self.model.evaluate(x_validation, y_validation, verbose=0)
        
        
    def __build_model(self):
        """Private method used to build the cnn classifier

        Returns:
            Sequential: A cnn model 
        """
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.dropout))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(learning_rate=self.learning_rate, rho=self.rho, epsilon=self.epsilon),
              metrics=['accuracy'])
        return model
