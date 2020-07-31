#Copyright (c) Microsoft Corporation. All rights reserved. 
#Licensed under the MIT License.

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Reshape, Dense, Dropout, Flatten, LeakyReLU, Conv2D, MaxPool2D, ZeroPadding2D, Conv2DTranspose, UpSampling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.datasets import mnist,cifar10
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
from scipy import stats
import warnings
import pandas as pd 
from privacygan.mnist.mnist_gan import MNIST_Discriminator, MNIST_Generator, MNIST_DiscriminatorPrivate
warnings.filterwarnings("ignore")


##########################################  GANs ######################################################################

def SimpGAN(X_train, generator = MNIST_Generator(), discriminator = MNIST_Discriminator(),
            randomDim=100, epochs=200, batchSize=128, optim = Adam(lr=0.0002, beta_1=0.5),
           verbose = 1, lSmooth = 0.9, SplitTF = False):

    # Combined network
    discriminator.trainable = False
    ganInput = Input(shape=(randomDim,))
    x = generator(ganInput)
    ganOutput = discriminator(x)
    gan = Model(inputs=ganInput, outputs=ganOutput)
    gan.compile(loss='binary_crossentropy', optimizer=optim)

    dLosses = []
    gLosses = []
    
    batchCount = X_train.shape[0] / batchSize
    print('Epochs:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', batchCount)

    for e in range(1, epochs+1):
        g_t = []
        d_t = []
        for i in range(int(batchCount)):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchSize)]

            # Generate fake MNIST images
            generatedImages = generator.predict(noise)
            # print np.shape(imageBatch), np.shape(generatedImages)
            X = np.concatenate([imageBatch, generatedImages])

            # Labels for generated and real data
            yDis = np.zeros(2*batchSize)
            # One-sided label smoothing
            yDis[:batchSize] = lSmooth

            # Train discriminator
            discriminator.trainable = True
            #dloss = discriminator.train_on_batch(X, yDis)
            if SplitTF:
                d_r = discriminator.train_on_batch(imageBatch, lSmooth*np.ones(batchSize))
                d_f = discriminator.train_on_batch(generatedImages,np.zeros(batchSize))    
                dloss = d_r + d_f
            else:
                dloss = discriminator.train_on_batch(X, yDis)

            discriminator.trainable = False


            # Train generator
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            yGen = np.ones(batchSize)
            gloss = gan.train_on_batch(noise, yGen)
            
            if verbose ==1:
                
                print(
                    'epoch = %d/%d, batch = %d/%d, d_loss=%.3f, g_loss=%.3f' % 
                    (e, epochs, i, batchCount, dloss, gloss),
                    100*' ',
                    end='\r'
                )
            
            d_t += [dloss]
            g_t += [gloss]

        # Store loss of most recent batch from this epoch
        dLosses.append(np.mean(d_t))
        gLosses.append(np.mean(g_t))
        
        if e%verbose == 0:
            print('epoch = %d/%d, d_loss=%.3f, g_loss=%.3f' % (e, epochs,  np.mean(d_t),np.mean(g_t)), 100*' ')
                
    return (generator, discriminator, dLosses, gLosses)



def TrainDiscriminator(X_train, y_train, discriminator = MNIST_DiscriminatorPrivate(OutSize = 2),
            randomDim=100, epochs=200, batchSize=128, optim = Adam(lr=0.0002, beta_1=0.5),
           verbose = 1):


    discriminator.fit(X_train, y_train,
          batch_size=batchSize,
          epochs=epochs,
          verbose=verbose,
          validation_data=(X_train, y_train))
    
        
    return (discriminator) 
    
    

def privGAN(X_train, generators = [MNIST_Generator(),MNIST_Generator()], 
            discriminators = [MNIST_Discriminator(),MNIST_Discriminator()],
            pDisc = MNIST_DiscriminatorPrivate(OutSize = 2), 
            randomDim=100, disc_epochs = 50, epochs=200, dp_delay = 100, 
            batchSize=128, optim = Adam(lr=0.0002, beta_1=0.5), verbose = 1, 
            lSmooth = 0.95, privacy_ratio = 1.0, SplitTF = False):
    
    
    #make sure the number of generators is the same as the number of discriminators 
    if len(generators) != len(discriminators):
        print('Different number of generators and discriminators')
        return()
    else:
        n_reps = len(generators)
        
    #throw error if n_reps = 1 
    if n_reps == 1:
        print('You cannot have only one generator-discriminator pair')
        return()
    
    
    X = []
    t = len(X_train)//n_reps
    y_train = []
    
    for i in range(n_reps):
        if i<n_reps-1:
            X += [X_train[i*t:(i+1)*t]]
            y_train += [i]*t
        else:
            X += [X_train[i*t:]]
            y_train += [i]*len(X_train[i*t:])
    
    y_train = np.array(y_train) + 0.0 
    
    pDisc2 = pDisc
        
    pDisc2.fit(X_train, y_train,
          batch_size=batchSize,
          epochs=disc_epochs,
          verbose=verbose,
          validation_data=(X_train, y_train))
    
    yp= np.argmax(pDisc2.predict(X_train), axis = 1)
    print('dp-Accuracy:',np.sum(y_train == yp)/len(yp))

    
    
    #define combined model 
    outputs = []    
    ganInput = Input(shape=(randomDim,))
    loss = ['binary_crossentropy']*n_reps + ['sparse_categorical_crossentropy']*n_reps
    Pout = []
    loss_weights = [1.0]*n_reps + [1.0*privacy_ratio]*n_reps
    
    pDisc2.trainable = False

    for i in range(n_reps):
        discriminators[i].trainable = False
        outputs += [discriminators[i](generators[i](ganInput))]
        Pout += [pDisc2(generators[i](ganInput))]
        
        
    #specify the combined GAN 
    outputs += Pout
    gan = Model(inputs = ganInput, outputs = outputs)       
    gan.compile(loss = loss, loss_weights = loss_weights, optimizer=optim)

            
    #Get batchcount
    batchCount = int(t // batchSize)
    print('Epochs:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', batchCount)
    
    dLosses = np.zeros((n_reps,epochs))
    dpLosses = np.zeros(epochs)
    gLosses = np.zeros(epochs)

    for e in range(epochs):
        d_t = np.zeros((n_reps,batchCount))
        dp_t = np.zeros(batchCount)
        g_t = np.zeros(batchCount)
        d_t3acc = np.zeros(batchCount)
        
        for i in range(batchCount):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            imageBatch = []
            generatedImages = []
            yDis2 = []
            yDis2f = []
            pDisc2.trainable = False
            
            
            for j in range(n_reps):
                imageBatch = X[j][np.random.randint(0, len(X[j]), size=batchSize)]
                generatedImages += [generators[j].predict(noise)]

                yDis = np.zeros(2*batchSize)
                yDis[:batchSize] = lSmooth
                discriminators[j].trainable = True
                
                if SplitTF:                    
                    d_r = discriminators[j].train_on_batch(imageBatch, lSmooth*np.ones(batchSize))
                    d_f = discriminators[j].train_on_batch(generatedImages[j],np.zeros(batchSize))
                    d_t[j,i] = d_r + d_f
                else:
                    X_temp = np.concatenate([imageBatch, generatedImages[j]])
                    d_t[j,i] = discriminators[j].train_on_batch(X_temp, yDis)
                    
                discriminators[j].trainable = False
                l = list(range(n_reps))
                del(l[j])
                yDis2 += [j]*batchSize
                yDis2f += [np.random.choice(l,(batchSize,))]
            
            yDis2 = np.array(yDis2)
            
            #Train privacy discriminator
            generatedImages = np.concatenate(generatedImages)

            
            if e >= dp_delay: 
                pDisc2.trainable = True
                dp_t[i] = pDisc2.train_on_batch(generatedImages, yDis2)
                pDisc2.trainable = False
                
            
            yGen = [np.ones(batchSize)]*n_reps + yDis2f
            
            #Train combined model
            g_t[i] = gan.train_on_batch(noise, yGen)[0]
            
            if verbose == 1:
                print(
                    'epoch = %d/%d, batch = %d/%d' % (e, epochs, i, batchCount),
                    100*' ',
                    end='\r'
                )

                      

        # Store loss of most recent batch from this epoch
        dLosses[:,e] = np.mean(d_t, axis = 1)
        dpLosses[e] = np.mean(dp_t)
        gLosses[e] = np.mean(g_t)
        
        if e%verbose == 0:
            print('epoch =',e)
            print('dLosses =', np.mean(d_t, axis = 1))
            print('dpLosses =', np.mean(dp_t))
            print('gLosses =', np.mean(g_t))
            yp= np.argmax(pDisc2.predict(generatedImages), axis = 1)
            print('dp-Accuracy:',np.sum(yDis2 == yp)/len(yp))
            
    return (generators, discriminators, pDisc2, dLosses, dpLosses, gLosses)


######################################### Ancillary functions ##########################################################



def DisplayImages(generator, randomDim = 100, NoImages = 100, figSize = (10,10), TargetShape = (28,28)):
    
    #check to see if the figure size is valid
    if (len(figSize)!=2) or (figSize[0]*figSize[1]<NoImages):
        print('Invalid Figure Size')
        return()
    
    #generate the synthetic images for a given generator
    noise = np.random.normal(0, 1, size=[NoImages, randomDim]) 
    generatedImages = generator.predict(noise)
    
    #re-shape the images 
    TargetShape = tuple([NoImages]+list(TargetShape))
    generatedImages = generatedImages.reshape(TargetShape)
    
    #plot the images 
    for i in range(generatedImages.shape[0]):
        plt.subplot(figSize[0],figSize[1], i+1)
        plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()

    
############################################ Attacks #####################################################################

def WBattack(X,X_comp, discriminator):
    
    Dat = np.concatenate([X, X_comp])
    p = discriminator.predict(Dat)
    In = np.argsort(-p[:,0])
    In = In[:len(X)]
    Accuracy = np.sum(1.*(In<len(X)))/len(X)
    print('White-box attack accuracy:',Accuracy)
    
    return(Accuracy)


def WBattack_priv(X,X_comp, discriminators):
    
    Dat = np.concatenate([X, X_comp])
    Pred = []
    
    for i in range(len(discriminators)):
        Pred += [discriminators[i].predict(Dat)[:,0]]
        
    
    p_mean = np.mean(Pred, axis = 0)
    p_max = np.max(Pred, axis = 0)
    
    In_mean = np.argsort(-p_mean)
    In_mean = In_mean[:len(X)]

    In_max = np.argsort(-p_max)
    In_max = In_max[:len(X)]
    
    Acc_max = np.sum(1.*(In_max<len(X)))/len(X)
    Acc_mean = np.sum(1.*(In_mean<len(X)))/len(X)
    
    print('White-box attack accuracy (max):',Acc_max)
    print('White-box attack accuracy (mean):',Acc_mean)
    
    return(Acc_max,Acc_mean)
        
    
def WBattack_TVD(X,X_comp, discriminator):
    
    n1, _ = np.histogram(discriminator.predict(X)[:,0], bins = 50, density = True, range = [0,1])
    n2, _ = np.histogram(discriminator.predict(X_comp)[:,0], bins = 50, density = True, range = [0,1])
    tvd = 0.5*np.linalg.norm(n1-n2,1)/50.0
    
    print('Total Variational Distance:',tvd)
    
    return(tvd)    
    
def WBattack_TVD_priv(X,X_comp, discriminators):
    
    tvd = []
    
    for i in range(len(discriminators)):
        n1, _= np.histogram(discriminators[i].predict(X)[:,0], bins = 50, density = True, range = [0,1])
        n2, _ = np.histogram(discriminators[i].predict(X_comp)[:,0], bins = 50, density = True, range = [0,1])
        tvd += [0.5*np.linalg.norm(n1-n2,1)/50.0]
    
    print('Total Variational Distance - max:',max(tvd))
    print('Total Variational Distance - mean:',np.mean(tvd))
    
    return(np.max(tvd),np.mean(tvd))   


def MC_eps_attack(X, X_comp, X_ho, generator, N = 100000, M = 100, n_pc = 40, reps = 10):
    
    #flatten images 
    if len(X.shape)==3:
        sh = X.shape[1]*X.shape[2]        
    elif len(X.shape)==2:
        sh = X.shape[1]
    else:
        sh = X.shape[1]*X.shape[2]*X.shape[3]
        
    X = np.reshape(X, (len(X),sh))
    X_comp = np.reshape(X_comp, (len(X_comp),sh))
    X_ho = np.reshape(X_ho, (len(X_ho),sh))
    
    #fit PCA
    pca = PCA(n_components=n_pc)
    pca.fit(X_ho)
        
    res = []
    
    for r in range(reps):
        
        #generate, flatten and dimensionality reduce a ton of synthetic samples 
        noise = np.random.normal(0, 1, size=[N, 100])
        X_fake = generator.predict(noise)
        X_fake = np.reshape(X_fake,(len(X_fake),sh))
        X_fake_dr = pca.transform(X_fake)
    
                
        idx1 = np.random.randint(len(X), size=M)
        idx2 = np.random.randint(len(X_comp), size=M)
    
        M_x = pca.transform(np.reshape(X[idx1,:],(len(X[idx1,:]),sh)))
        M_xc = pca.transform(np.reshape(X_comp[idx1,:],(len(X_comp[idx1,:]),sh)))
        
        min_x = []
        min_xc = []

        #calculate median epsilon 
        for i in range(M):
            temp_x = np.tile(M_x[i,:],(len(X_fake_dr),1))
            temp_xc = np.tile(M_xc[i,:],(len(X_fake_dr),1))

            D_x = np.sqrt(np.sum((temp_x-X_fake_dr)**2,axis=1))
            D_xc = np.sqrt(np.sum((temp_xc-X_fake_dr)**2,axis=1))

            min_x += [np.min(D_x)]
            min_xc += [np.min(D_xc)]
            
        eps = np.median(min_x + min_xc)
            
        s_x = []
        s_xc = []
        
        #estimate the integral
        for i in range(M):
            temp_x = np.tile(M_x[i,:],(len(X_fake_dr),1))
            temp_xc = np.tile(M_xc[i,:],(len(X_fake_dr),1))

            D_x = np.sqrt(np.sum((temp_x-X_fake_dr)**2,axis=1))
            D_xc = np.sqrt(np.sum((temp_xc-X_fake_dr)**2,axis=1))

            s_x += [np.sum(D_x <= eps)/len(X_fake_dr)]
            s_xc += [np.sum(D_xc <= eps)/len(X_fake_dr)]
            
        s_x_xc = np.array(s_x + s_xc)
        In = np.argsort(-s_x_xc)[:M]


        if np.sum(In<M)>= 0.5*M:
            res += [1]
        else:
            res += [0]
            
    
    return(np.mean(res))
            
    
    

def MC_eps_attack_priv(X, X_comp, X_ho, generators, N = 100000, M = 100, n_pc = 40, reps = 10):
    
    #flatten images 
    if len(X.shape)==3:
        sh = X.shape[1]*X.shape[2]        
    elif len(X.shape)==2:
        sh = X.shape[1]
    else:
        sh = X.shape[1]*X.shape[2]*X.shape[3]
        
    X = np.reshape(X, (len(X),sh))
    X_comp = np.reshape(X_comp, (len(X_comp),sh))
    X_ho = np.reshape(X_ho, (len(X_ho),sh))
    
    #fit PCA
    pca = PCA(n_components=n_pc)
    pca.fit(X_ho)
        
    res = []
    
    for r in range(reps):
        
        #generate, flatten and dimensionality reduce a ton of synthetic samples 
        n_g = len(generators)
        X_fake_dr  = []
        for j in range(n_g):
            noise = np.random.normal(0, 1, size=[int(N/n_g), 100])
            X_fake = generators[j].predict(noise)
            X_fake = np.reshape(X_fake,(len(X_fake),sh))
            X_fake_dr += [pca.transform(X_fake)]
            
        X_fake_dr = np.vstack(X_fake_dr)

        
        idx1 = np.random.randint(len(X), size=M)
        idx2 = np.random.randint(len(X_comp), size=M)
    
        M_x = pca.transform(np.reshape(X[idx1,:],(len(X[idx1,:]),sh)))
        M_xc = pca.transform(np.reshape(X_comp[idx1,:],(len(X_comp[idx1,:]),sh)))
        
        min_x = []
        min_xc = []

        #calculate median epsilon 
        for i in range(M):
            temp_x = np.tile(M_x[i,:],(len(X_fake_dr),1))
            temp_xc = np.tile(M_xc[i,:],(len(X_fake_dr),1))

            D_x = np.sqrt(np.sum((temp_x-X_fake_dr)**2,axis=1))
            D_xc = np.sqrt(np.sum((temp_xc-X_fake_dr)**2,axis=1))

            min_x += [np.min(D_x)]
            min_xc += [np.min(D_xc)]
            
        eps = np.median(min_x + min_xc)
            
        s_x = []
        s_xc = []
        
        #estimate the integral
        for i in range(M):
            temp_x = np.tile(M_x[i,:],(len(X_fake_dr),1))
            temp_xc = np.tile(M_xc[i,:],(len(X_fake_dr),1))

            D_x = np.sqrt(np.sum((temp_x-X_fake_dr)**2,axis=1))
            D_xc = np.sqrt(np.sum((temp_xc-X_fake_dr)**2,axis=1))

            s_x += [np.sum(D_x <= eps)/len(X_fake_dr)]
            s_xc += [np.sum(D_xc <= eps)/len(X_fake_dr)]
            
        s_x_xc = np.array(s_x + s_xc)
        In = np.argsort(-s_x_xc)[:M]


        if np.sum(In<M)>= 0.5*M:
            res += [1]
        else:
            res += [0]
            
    
    return(np.mean(res))
