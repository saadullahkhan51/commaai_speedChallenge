## this version starts training, Loss skyrockets
## TODO
## -change range to xtrain.size 
## -create test data
## -create predictor for new data


import pandas as pd
import Model
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

data_size = 20400
batch_size = 16
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


imgBatch = np.zeros((len(xTrain), 66, 220, 3))
labelBatch = np.zeros((len(yTrain)))

# prints for debugging
print(xTest[0])
print(xTrain[-1])
img = cv2.imread(X[0])
print(img.shape)
imgBatch[0] = img
print(imgBatch[0].shape)

def GenerateTrainData():
    for i in range (0, len(xTrain)):
        try:
            img = cv2.imread(xTrain[i])
            imgBatch[i] = img
            labelBatch[i] = yTrain[i]
            #print(imgBatch[i].shape)
        except Exception as e:
            pass
GenerateTrainData()

print(imgBatch.shape)
print(labelBatch.shape)

myModel = Model.NVIDIA_model()
history = myModel.fit(imgBatch, labelBatch ,batch_size= batch_size, epochs= 30)
print(history)