## this version starts training, Loss skyrockets
## TODO
## -change range to xtrain.size 
## -create test data
## -create predictor for new data


import pandas as pd
import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pickle

data_size = 20400
batch_size = 32
steps_per_epoch = data_size/ batch_size
imgdir = "D:/venv/PyScripts/speedchallenge-master/data/data_preprocessed_dense" 

data = pd.read_csv("./preprocessed_dense.csv", header = None)

# X contains the image dirs, Y contains labels
X = data.iloc[:, 0].values
Y = data.iloc[:, 2].values

# splitting for test and train
xTrain = X[0:16320]
yTrain = Y[0:16320]
xTest = X[xTrain.size : X.size]
yTest = X[yTrain.size : Y.size]


imgBatch = np.zeros((len(X), 66, 220, 3))
labelBatch = np.zeros((len(Y)))

def GenerateTrainData():
    for i in range (0, len(X)):
        try:
            img = cv2.imread(X[i])
            imgBatch[i] = img
            labelBatch[i] = Y[i]
            #print(imgBatch[i].shape)
        except Exception as e:
            pass

#GenerateTrainData()
print(imgBatch.shape)
print(labelBatch.shape)



if __name__ == "__main__":
    filepath = './model_weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    GenerateTrainData()
    print(imgBatch.shape)
    print(labelBatch.shape)
    myModel = Model.Pooled_model()
    earlyStopping = EarlyStopping(monitor='val_loss', 
                              patience=2, 
                              verbose=1, 
                              min_delta = 0.23,
                              mode='min',)
    modelCheckpoint = ModelCheckpoint(filepath = filepath, 
                                      monitor = 'val_loss', 
                                      save_best_only = True, 
                                      mode = 'min', 
                                      verbose = 1,
                                     save_weights_only = True)
    callbacks_list = [modelCheckpoint, earlyStopping]
    history = myModel.fit(imgBatch, labelBatch ,batch_size= batch_size, epochs= 10, validation_split= 0.1, callbacks= callbacks_list)
    score = myModel.evaluate(imgBatch, labelBathc, verbose = 0)
    print(score)
    print(history)
    