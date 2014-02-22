import numpy as np
import cv2
import pdb
import math
import glob
import os
from PIL import Image

INF = 1e10

# location of pages
page_dir = '/Users/karthikv/Dropbox/Shared/CS231A/scoryst-project/cs221-exams-better'
pages = range(0, 6)

prev_dir = os.getcwd()
os.chdir(page_dir)

blank_image_names = glob.glob('*blank.jpg')
blank_images = map(lambda page: cv2.imread(page, 0), blank_image_names)

page_image_names = {}
page_images = {}
all_page_images = []

print 'Reading images...'
# read images for each page
for page in pages:
  page_image_names[page] = glob.glob('*%d.jpg' % page)
  page_images[page] = map(lambda page: cv2.imread(page, 0), page_image_names[page])
  all_page_images = all_page_images + page_images[page]

os.chdir(prev_dir)

# find lowest dimensions
min_width = reduce(lambda memo, image: min(memo, image.shape[1]), all_page_images, INF)
min_height = reduce(lambda memo, image: min(memo, image.shape[0]), all_page_images, INF)
min_dimension = (min_width, min_height)

print 'Resizing images...'
# resize images to lowest dimensions
blank_images = map(lambda image: cv2.resize(image, min_dimension), blank_images)
for page in pages:
  page_images[page] = map(lambda image: cv2.resize(image, min_dimension), page_images[page])

detector = cv2.FeatureDetector_create('SURF')
descriptor = cv2.DescriptorExtractor_create('SURF')

blank_keypoints = {}
print 'Computing keypoints for blank pages...'

# compute blank keypoints
for blank_page in pages:
  keypoints = detector.detect(blank_images[blank_page])
  blank_keypoints[blank_page] = descriptor.compute(blank_images[blank_page], keypoints)

page_keypoints = {}
print 'Computing keypoints for pages...'

# compute page keypoints
for page in pages:
  keypoints = detector.detect(page_images[page][0])
  page_keypoints[page] = map(lambda image: descriptor.compute(image, keypoints), page_images[page])

matcher = cv2.DescriptorMatcher_create('FlannBased')

print 'Matching keypoints...'
total_positive = 0
total_negative = 0

true_positive = 0
true_negative = 0

for blank_page in pages:
  blank_kp, blank_descriptor = blank_keypoints[blank_page]

  for page in pages:
    for page_kp, page_descriptor in page_keypoints[page]:
      # match the keypoints
      matches = matcher.match(blank_descriptor, page_descriptor)
      top_matches = sorted(matches, key=lambda match: match.distance)[0:10]

      blank_points = np.array(map(lambda match: [blank_kp[match.queryIdx].pt], top_matches))
      page_points = np.array(map(lambda match: [page_kp[match.trainIdx].pt], top_matches))

      # transform = cv2.estimateRigidTransform(blank_images[blank_page], page_images[page][0], fullAffine=False)
      transform = cv2.estimateRigidTransform(np.float32(blank_points), np.float32(page_points), fullAffine=False)

      match = True
      message = ''

      if transform == None:
        match = False
        message = 'no transform'
      else:
        scale = (transform[0][0] ** 2 + transform[0][1] ** 2) ** 0.5
        transform_no_scale = transform[0:2, 0:2] / scale
        angle = math.acos(transform_no_scale[0][0])

        if angle > 0.4:
          match = False
          message = 'angle %g too large' % angle
        elif scale < 0.5:
          match = False
          message = 'scale %g too small' % scale
        elif scale > 3:
          match = False
          message = 'scale %g too large' % scale
        else:
          match = True

      if page == blank_page:
        total_positive += 1
      else:
        total_negative += 1

      if match:
        if page == blank_page:
          true_positive += 1
          status = 'CORRECT'
        else:
          status = 'WRONG'

        print '[%s] page %d matches blank page %d!' % (status, page, blank_page)
      else:
        if page == blank_page:
          status = 'WRONG'
        else:
          true_negative += 1
          status = 'CORRECT'

        print '[%s] {%s} page %d does not match blank page %d' % (status, message, page, blank_page)

print 'Total positive: %g, Total negative: %g' % (total_positive, total_negative)
print 'True positive: %g (%g%%), True negative: %g (%g%%)' % (true_positive,
  float(true_positive) * 100 / total_positive, true_negative,
  float(true_negative) * 100 / total_negative)

# transformed = cv2.warpAffine(blank_pages[1], transform, min_dimension)
# 
# image_out = Image.fromarray(transformed.astype('uint8'))
# image_out.save('transformed_out.jpg')
# 
# image_out = Image.fromarray(test_pages[1].astype('uint8'))
# image_out.save('actual.jpg')


# 
# 
# 
# 
# height, width = blank_pages[0].shape
# transformed = cv2.warpAffine(images[0], transform, (width, height))
# 
# cv2.imshow('second_image', images[1])
# cv2.imshow('transformed_first_image', transformed)
# 
# transform[0:2, 0:2] /= ((transform[0][0] ** 2 + transform[0][1] ** 2) ** 0.5)
# 
# while True:
#   # handle events
#   k = cv2.waitKey(10)
# 
#   if k == 0x1b: # quit on esc
#     print 'ESC pressed. Exiting ...'
#     break
