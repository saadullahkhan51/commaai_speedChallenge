import cv2
import numpy as np 
import os
import matplotlib.pyplot as plt 
import csv

inputPath = "./speedchallenge-master/data/train.mp4" #path to input video
labels = "./speedchallenge-master/data/train.txt"
outputPath = "./speedchallenge-master/data/data_preprocessed_dense/" 

# if path does no exist it is created
if not os.path.exists(outputPath):
    os.mkdir(outputPath)

def CropImage(frame):
    frame = frame[100:380, :]
    image = cv2.resize(frame, (220, 66), interpolation=cv2.INTER_AREA)
    return image 
    # dims for frame with sky and dash cropped frame[100:380, :] - can be narrowed in x dim aswell

def LabelPlot(labelPath):
    # plots the labels as they are using matplotlib
    labelPath = np.loadtxt(labelPath)
    plt.plot(labelPath)
    plt.show()
    

def FrameLabelPair(videoPath, labelPath):
    labels = np.loadtxt(labelPath)
    label = []
    cap = cv2.VideoCapture(videoPath)
    ret, frame = cap.read()
    frame = CropImage(frame)
    prvs = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame)
    hsv[...,1] = 255
    count = 0
    with open('preprocessed_dense.csv', 'w') as file:
        writer = csv.writer(file)

        while(cap.isOpened()):
            label.append(float((labels[count] + labels[count+1])/2))
            ret, frame2 = cap.read()
            frame2 = CropImage(frame2)
            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

            #cv2.imshow('DenseFlow',rgb)
            prvs = next
            if not ret:
                break
            img = outputPath + "img" + str(count)+'.jpg'
            cv2.imwrite(img, rgb)
            writer.writerow([img, count, label[count]])
            count += 1
            if cv2.waitKey(20) == 27:
                break
    print("completed")
    file.close()
    cap.release()
    cv2.destroyAllWindows()
    # need to decide the order of function calls- do we crop and apply LK after the pairs are created or create pairs with the processed frames
    # Pairs the frame to its corresponding label for training
    # add assertion for frame-label count equivalency

if __name__ == "__main__":
    #LabelPlot(labels)
    FrameLabelPair(inputPath, labels)

