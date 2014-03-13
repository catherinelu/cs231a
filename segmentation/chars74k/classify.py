import os
import glob
import collections
import pickle
import cv2

from skimage import feature as image_feature
from sklearn import svm, neighbors


# how many training images we have for each letter
NUM_IMAGES_PER_LETTER = 55

# where images are located
BASE_IMAGE_DIR = './cropped'

# letters correspond to directories where images are stored
LETTERS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

# file to store the classifier in
CLASSIFIER_FILE_NAME = 'classifier-knn'


def read_training_images():
  """
  Reads in training images in both color and grayscale. Returns a tuple of
  (images, gray_images, image_labels).
  """
  gray_images = collections.defaultdict(list)
  images = collections.defaultdict(list)

  for letter in LETTERS:
    # find correct directory for lowercase/uppercase character images
    if letter.islower():
      image_dir = '%s/%s-lower' % (BASE_IMAGE_DIR, letter)
    else:
      image_dir = '%s/%s' % (BASE_IMAGE_DIR, letter)

    # read in letter images in both color and grayscale
    for i in range(1, NUM_IMAGES_PER_LETTER + 1):
      image_path = '%s/%s.png' % (image_dir, i)
      image = cv2.imread(image_path)
      images[letter].append(image)

      gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      gray_images[letter].append(gray_image)

  return images, gray_images


def compute_features(image):
  """ Returns the import features within the given image. """
  max_y, max_x = image.shape
  features = image_feature.hog(image, pixels_per_cell=(max_x / 3, max_y / 3),
    cells_per_block=(2, 2))
  return features


def train(all_images):
  """ Trains and returns a classifier on the given images. """
  training_vectors = []
  training_labels = []

  for letter in LETTERS:
    letter_images = all_images[letter]

    # determine training vector and label for each image
    for image in letter_images:
      features = compute_features(image)

      training_vectors.append(features)
      training_labels.append(ord(letter))


  # train an SVM classifier
  # classifier = svm.SVC(gamma=0.001, probability=True)
  # classifier.fit(training_vectors, training_labels)

  # train a KNN classifier
  classifier = neighbors.KNeighborsClassifier(5)
  classifier.fit(training_vectors, training_labels)
  return classifier


def store_classifier(classifier):
  """ Stores the given classifier so it doesn't have to be re-trained. """
  with open(CLASSIFIER_FILE_NAME, 'w') as handle:
    pickle.dump(classifier, handle)


def read_classifier():
  """
  Reads and returns the stored classifier. Returns None if no classifier could
  be found.
  """
  try:
    with open(CLASSIFIER_FILE_NAME, 'r') as handle:
      classifier = pickle.load(handle)
  except IOError, pickle.UnpicklingError:
    classifier = None

  return classifier


if __name__ == '__main__':
  print 'Reading training images...'
  images, gray_images = read_training_images()

  print 'Training...'
  classifier = train(gray_images)

  print 'Storing classifier...'
  store_classifier(classifier)
