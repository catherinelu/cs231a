from collections import Counter
import copy
import cv2
import numpy as np
import pickle
from sklearn import svm, metrics
import pdb

def unskew_training_data(labels, training_data):
    counter = Counter()
    for l in labels:
        counter[l] += 1
    highest_count = counter.most_common(1)[0][1]
    unskewed_training_data = copy.deepcopy(training_data)
    unskewed_labels = copy.deepcopy(labels)
    for key in counter:
        while True:
            for i, features in enumerate(training_data):
                if labels[i] == key and counter[key] < highest_count:
                    unskewed_training_data.append(training_data[i])
                    unskewed_labels.append(labels[i])
                    counter[key] += 1
                elif counter[key] >= highest_count:
                    break
            if counter[key] > highest_count:
                exit(1)
            elif counter[key] == highest_count:
                break
    return unskewed_labels, unskewed_training_data



#######   training part    ############### 
f = open('training_samples')
training_data = pickle.load(f)

labels = [ord(d[1]) for d in training_data]
training_data = [d[0][0] for d in training_data]

labels, training_data = unskew_training_data(labels, training_data)

print labels

svm_classifier = svm.SVC(gamma=0.001, probability=True)  # Eventually should set probability=True
svm_classifier.fit(training_data, labels)

############################# testing part  #########################

im = cv2.imread('images/adam.png')
out = np.zeros(im.shape,np.uint8)
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

image, contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

adam_contours = []

for cnt in contours:
    if cv2.contourArea(cnt)>50:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  h>28:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            roismall = roismall.reshape((1,100))
            predicted = svm_classifier.predict(roismall[0])
            print svm_classifier.predict_proba(roismall[0])
            adam_contours.append(roismall[0])
            char = chr(predicted[0])
            cv2.putText(out,char,(x,y+h),0,1,(0,255,0))

pdb.set_trace()

cv2.imshow('im',im)
cv2.imshow('out',out)
cv2.waitKey(0)


# """
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
# """

# import pylab as pl
# from sklearn import svm, metrics
# import pickle

# # Create a classifier: a support vector classifier
# classifier = svm.SVC(gamma=0.001, probability=True)

# f = open("characterdata24px", "r")
# trainDataX, trainDataY, testDataX, testDataY = pickle.load(f)

# classifier.fit(trainDataX, trainDataY)

# # To pickle the SVM classifier
# svm_f = open("svm24px", "w")
# pickle.dump(classifier, svm_f)

# expected = testDataY
# predicted = classifier.predict(testDataX)

# print classifier.score(testDataX, testDataY)

# print("Classification report for classifier %s:\n%s\n"
#       % (classifier, metrics.classification_report(expected, predicted)))