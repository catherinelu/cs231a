import numpy as np
import cv2
import classify
import time
import collections
import fnmatch


def crop_image(image):
  shape_y, shape_x = image.shape
  horizontal_sum = image.sum(axis=1)

  image = image[horizontal_sum != 255 * shape_x]
  # for i in range(shape_y):
  #   if not horizontal_sum[i] == 255 * shape_x:
  #     break

  # print i
  return image


def split_into_vertical_strips(image):
  shape_y, shape_x = image.shape
  veritcal_sum = image.sum(axis=0)

  last_breakpoint = 0
  was_last_column_white = True

  splits = []

  for x in range(shape_x):
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

image = cv2.imread('ben.png')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
splits = split_into_vertical_strips(image_gray)

new_splits = []
for i in xrange(len(splits)):
  split = cv2.GaussianBlur(splits[i], (5, 5), 8)
  value, split = cv2.threshold(split, 200, 255, cv2.THRESH_BINARY)

  split = crop_image(split)
  if split.size > 0 and split.shape[0] >= 9 and split.shape[1] >= 9:
    new_splits.append(split)

splits = new_splits

# for i, split in enumerate(splits):
#   cv2.destroyAllWindows()
#   cv2.imshow('image', split)
#   time.sleep(1)

classifier = classify.read_classifier()
high_accuracy = [False for split in splits]
match_string = ''

for i, split in enumerate(splits):
  features = classify.compute_features(split)
  prediction = chr(classifier.predict(features)[0])
  probabilities = classifier.predict_proba(features)

  if probabilities.max() > 0.9:
    print 'split has high accuracy', prediction
    high_accuracy[i] = True
    cv2.destroyAllWindows()
    cv2.imshow('split', split)
    time.sleep(0.5)

    match_string += prediction

    if len(match_string) >= 4:
      match_string += '*'
      break
  else:
    if len(match_string) == 0 or not match_string[-1] == '*':
      match_string += '*'

print 'match string is', match_string
with open('roster', 'r') as handle:
  roster = handle.readlines()

roster = map(lambda name: name.strip(), roster)
filtered_roster = filter(lambda name: fnmatch.fnmatch(name.lower(), match_string.lower()), roster)
filtered_roster = map(lambda name: (name.replace(' ', ''), 0, 0), filtered_roster)

for i, split in enumerate(splits):
  print 'filtered roster', np.array(filtered_roster)
  old_indexes = map(lambda data: data[1], filtered_roster)
  shape_y, shape_x = split.shape
  start = 0

  if high_accuracy[i]:
    filtered_roster = map(lambda data: (data[0], data[1] + 1, data[2] + 1), filtered_roster)
  else:
    while start < shape_x:
      end = start + 30
      possible_chars = set()

      while end <= shape_x:
        image = split[0:shape_y, start:end]
        image = crop_image(image)

        if not image.size == 0:
          features = classify.compute_features(image)
          prediction = chr(classifier.predict(features)[0])
          probabilities = classifier.predict_proba(features)

          max_probability = probabilities.max()
          if max_probability >= 0.4:
            possible_chars.add(prediction.lower())

          # print end, max_probability, prediction
          print end, prediction

        if end == shape_x:
          end = shape_x + 1
        else:
          end = min(end + 10, shape_x)

      print 'possible chars', possible_chars
      new_filtered_roster = []
      for j, data in enumerate(filtered_roster):
        name, index, votes = data

        if (not index == len(name)) and (name[index].lower() in possible_chars):
          new_filtered_roster.append((name, index + 1, votes + 1))
        else:
          new_filtered_roster.append((name, index, votes))

      filtered_roster = new_filtered_roster
      start = start + 10

  for i, old_index in enumerate(old_indexes):
    name, index, votes = filtered_roster[i]

    if index == len(name):
      continue

    if index == old_index:
      filtered_roster[i] = (name, index + 1, votes)

scores = map(lambda data: float(data[2]) / (len(data[0])), filtered_roster)
print np.array([filtered_roster, scores])
max_index = np.argmax(scores)
print filtered_roster[max_index], scores[max_index]


# for split in splits:
#   shape_y, shape_x = split.shape
#   start = 0
# 
#   while start < shape_x:
#     end = start + 10
#     best_ends = collections.defaultdict(list)
# 
#     while end <= shape_x:
#       image = split[0:shape_y, start:end]
#       image = crop_image(image)
# 
#       if not image.size == 0:
#         features = classify.compute_features(image)
#         prediction = chr(classifier.predict(features)[0])
#         probabilities = classifier.predict_proba(features)
# 
#         max_probability = probabilities.max()
#         if abs(max_probability - 1) < 0.1:
#           best_ends[prediction].append(end)
# 
#         print end, max_probability, prediction
# 
#       if end == shape_x:
#         end = shape_x + 1
#       else:
#         end = min(end + 10, shape_x)
# 
#     best_char = None
#     for char in best_ends.keys():
#       if best_char == None or len(best_ends[char]) > len(best_ends[best_char]):
#         best_char = char
# 
#     if best_char == None:
#       start += 10
#       print 'no letter; next start', start
#     else:
#       start = np.max(best_ends[best_char])
#       print 'got letter', best_char, 'next start', start
# 
#   print 'done with split!'
