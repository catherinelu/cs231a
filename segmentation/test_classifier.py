import os
import glob
import sys
import cv2
import classify
import time


def classify_jpegs():
  """ Reads and classifies each JPEG in a user-provided directory. """
  args = sys.argv

  # ensure we have the right number of arguments
  if not len(args) == 2:
    print 'Usage: %s [directory]' % (args[0])
    print 'Classifies each JPEG image in the given directory.'
    return

  classifier = classify.read_classifier()

  directory = args[1]
  old_cwd = os.getcwd()
  os.chdir(directory)

  # find all images in directory
  image_paths = glob.glob('*.jpg') + glob.glob('*.jpeg') + glob.glob('*.png')

  # read and classify each image
  for image_path in image_paths:
    image = cv2.imread(image_path)
    features = classify.compute_features(image, preprocess=True)

    prediction = classifier.predict(features)[0]
    print image_path, chr(prediction)

  os.chdir(old_cwd)


if __name__ == '__main__':
  classify_jpegs()
