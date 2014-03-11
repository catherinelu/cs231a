import cv2
import fnmatch, os
import pickle
import random
import sys
import numpy as np


PICKLED_TRAINING_SAMPLES_PATH = 'training_samples'
SENTENCE_IMAGE_PATH = '/Users/catherinelu/Dropbox/CS231A/scoryst-project/names/data/sentences/a03/a03-059/'

def process_image(filename):
    im = cv2.imread(filename)

    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

    #################      Now finding Contours         ###################

    images,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt)>50:
            [x,y,w,h] = cv2.boundingRect(cnt)

            if  h>28:
                im3 = im.copy()
                cv2.rectangle(im3,(x,y),(x+w,y+h),(0, 0, 255),2)
                roi = thresh[y:y+h,x:x+w]
                roismall = cv2.resize(roi,(10,10))
                cv2.imshow('norm',im3)
                key = cv2.waitKey(0)

                if key == 27:  # escape to stop
                    return False
                elif key == 32:  # space to skip
                    continue
                else:
                    sample = roismall.reshape((1,100))
                    training_samples.append((sample, chr(key)))
    return True


# Array of tuples, where first entry contains features and second entry is the
# label. Loads from previous version if there have already been samples.
training_samples = []
if os.path.isfile(PICKLED_TRAINING_SAMPLES_PATH):
    f = open(PICKLED_TRAINING_SAMPLES_PATH)
    training_samples = pickle.load(f)

should_break = False
for root, dirnames, filenames in os.walk(SENTENCE_IMAGE_PATH):
    for filename in fnmatch.filter(filenames, '*.png'):
        if not process_image(os.path.join(root, filename)):
            should_break = True
            break
    if should_break:
        break


print "training complete"

f = open('training_samples', 'w')
pickle.dump(training_samples, f)
