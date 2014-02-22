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

# resize images to lowest dimensions
blank_images = map(lambda image: cv2.resize(image, min_dimension), blank_images)
for page in pages:
  page_images[page] = map(lambda image: cv2.resize(image, min_dimension), page_images[page])

for blank_page in [2]:
  for page in pages:
    print 'blank image:', blank_image_names[blank_page], 'page image:', page_image_names[page][0]
    transform = cv2.estimateRigidTransform(blank_images[blank_page], page_images[page][0], fullAffine=False)

    if transform == None:
      print "page %d does not match with page %d" % (page, blank_page)
    else:
      scale = (transform[0][0] ** 2 + transform[0][1] ** 2) ** 0.5
      transform_no_scale = transform[0:2, 0:2] / scale
      angle = math.acos(transform_no_scale[0][0])
      translation = transform[:, 2]

      print 'scale', scale
      print 'angle', angle
      print 'translation', translation


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
while True:
  # handle events
  k = cv2.waitKey(10)

  if k == 0x1b: # quit on esc
    print 'ESC pressed. Exiting ...'
    break
