from keras.models import Model, Sequential
#from keras.layers.convolutional import Conv3D
#from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dense, Activation, Conv2D, ELU, Flatten, Dropout, Lambda, MaxPooling2D
from keras.optimizers import Nadam
import numpy as np

# Embedding
maxFeatures = 20000
maxlen = 100
embeddingSize = 128

# Convolution
kernelSize = 5
filters = 64
poolSize = 4

# LSTM
lstmOutputSize = 70

# Training hyper parameters
batchSize = 16
epochs = 20

frameShape = (66, 220, 3)

# model from NVIDIA's End to End paper
def NVIDIA_model():
    
    model = Sequential()
    # normalization    
    # perform custom normalization before lambda layer in network
    model.add(Lambda(lambda x: x/ 127.5 - 1, input_shape = frameShape))

    model.add(Conv2D(24, (5, 5), 
                            strides=(2,2), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv1'))
    
    
    model.add(ELU())    
    model.add(Conv2D(36, (5, 5), 
                            strides=(2,2), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv2'))
    
    model.add(ELU())    
    model.add(Conv2D(48, (5, 5), 
                            strides=(2,2), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv3'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), 
                            strides = (1,1), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv4'))
    
    model.add(ELU())              
    model.add(Conv2D(64, (3, 3), 
                            strides= (1,1), 
                            padding = 'valid',
                            kernel_initializer = 'he_normal',
                            name = 'conv5'))
              
              
    model.add(Flatten(name = 'flatten'))
    model.add(ELU())
    model.add(Dense(100, kernel_initializer = 'he_normal', name = 'fc1'))
    model.add(ELU())
    model.add(Dense(50, kernel_initializer = 'he_normal', name = 'fc2'))
    model.add(ELU())
    model.add(Dense(10, kernel_initializer = 'he_normal', name = 'fc3'))
    model.add(ELU())
    
    # we do not put activation at the end because we want the exact output
    model.add(Dense(1, name = 'output', kernel_initializer = 'he_normal'))
    
    adam = Nadam()
    model.compile(optimizer = adam, loss = 'mse', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    model = NVIDIA_model()