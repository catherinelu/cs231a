import numpy as np
import cv2
import classify
import time
import collections
import fnmatch
import os
import glob
import roster as roster_module


def crop_image(image):
  shape_y, shape_x = image.shape
  horizontal_sum = image.sum(axis=1)

  for y in xrange(shape_y):
    if not horizontal_sum[y] == 255 * shape_x:
      break

  min_y = y

  for y in reversed(range(shape_y)):
    if not horizontal_sum[y] == 255 * shape_x:
      break

  max_y = y

  return image[min_y:max_y, 0:shape_x]


def split_into_vertical_strips(image):
  shape_y, shape_x = image.shape
  veritcal_sum = image.sum(axis=0)

  last_breakpoint = 0
  was_last_column_white = True

  splits = []

  for x in xrange(shape_x):
    # is this column filled with white pixels?
    if veritcal_sum[x] > 253 * shape_y:
      if not was_last_column_white:
        splits.append(image[0:shape_y, last_breakpoint:x])
        last_breakpoint = x
      was_last_column_white = True
    else:
      if was_last_column_white:
        # splits.append(image[0:shape_y, last_breakpoint:x])
        last_breakpoint = x
      was_last_column_white = False

  return splits

def edit_distance(s1, s2):
  COST_INSERTION = 1
  COST_DELETION = 1
  COST_REPLACEMENT = 1

  # distances[i][j] is the edit distance to convert the first i characters of s1
  # to the first j characters of s2
  distances = [[None for j in xrange(len(s2) + 1)] for i in xrange(len(s1) + 1)]

  # converting i characters of s1 to 0 characters of s2 takes i deletions
  for i in xrange(len(s1) + 1):
    distances[i][0] = i * COST_DELETION

  # converting 0 characters of s1 to j characters of s2 takes j insertions
  for j in xrange(len(s2) + 1):
    distances[0][j] = j * COST_INSERTION

  for i in xrange(1, len(s1) + 1):
    for j in xrange(1, len(s2) + 1):
      distances[i][j] = min(
        # convert i characters of s1 to j - 1 characters of s2; insert one additional
        distances[i][j - 1] + COST_INSERTION,

        # convert i - 1 characters of s1 to j characters of s2; delete i-th character
        distances[i - 1][j] + COST_DELETION,

        # convert i - 1 characters of s1 to j - 1 characters of s2; replace last
        # character if it's different
        distances[i - 1][j - 1] + (s1[i - 1] != s2[j - 1]) * COST_REPLACEMENT
      )

  return distances[-1][-1]


if __name__ == '__main__':
  classifier = classify.read_classifier()
  with open('roster', 'r') as handle:
    roster = handle.readlines()

  roster = map(lambda name: name.strip().lower().replace(' ', ''), roster)

  def find_closest_roster_match(name):
    min_dist = 1000
    best_name = None

    for roster_name in roster:
      dist = edit_distance(name, roster_name)
      if dist < min_dist:
        best_name = roster_name
        min_dist = dist

    return best_name

  os.chdir('./segmentable-names')
  names = glob.glob('*.png')

  num_total = 0
  num_correct = 0

  for name in names:
    image = cv2.imread(name)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    splits = split_into_vertical_strips(image_gray)

    new_splits = []
    for i in xrange(len(splits)):
      split = cv2.GaussianBlur(splits[i], (5, 5), 7)
      value, split = cv2.threshold(split, 200, 255, cv2.THRESH_BINARY)

      split = crop_image(split)
      if split.size > 0 and split.shape[0] >= 9 and split.shape[1] >= 9:
        new_splits.append(split)

    splits = new_splits

    match_string = ''
    for i, split in enumerate(splits):
      features = classify.compute_features(split)
      prediction = chr(classifier.predict(features)[0])
      probabilities = classifier.predict_proba(features)

      print probabilities.max(), prediction
      match_string += prediction

      # cv2.destroyAllWindows()
      # cv2.imshow('split', split)
      # time.sleep(0.3)

    name = name[:-4].strip().lower().replace('-', '')
    match_string = match_string.lower()
    guessed_name = find_closest_roster_match(match_string)

    print 'match string for', name, 'is', match_string, guessed_name, edit_distance(match_string, name), edit_distance(guessed_name, name)

    if guessed_name == name:
      num_correct += 1
    num_total += 1

  print 'Accuracy: %g/%g = %g' % (num_correct, num_total, float(num_correct) / num_total)

# with open('roster', 'r') as handle:
#   roster = handle.readlines()
# 
# roster = map(lambda name: name.strip(), roster)
# filtered_roster = filter(lambda name: fnmatch.fnmatch(name.lower(), match_string.lower()), roster)
# filtered_roster = map(lambda name: (name.replace(' ', ''), 0, 0), filtered_roster)
