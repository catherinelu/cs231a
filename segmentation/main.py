import cv2
import crop_name, classify, roster, segment
import math
import pdb

IMAGE_DIRECTORY = 'exams2-test'

classifier = classify.read_classifier()

# crop = crop_name.CropName(IMAGE_DIRECTORY)
# cropped_images, correct_names = crop.run()

cropped_images = [cv2.imread('cheng-chen.png', 0)]
correct_names = ['Bonnie McLindon']

for i, cropped_image in enumerate(cropped_images):
    cropped_image = cv2.GaussianBlur(cropped_image, (3, 3), 3)
    value, cropped_image = cv2.threshold(cropped_image, 200, 255, cv2.THRESH_BINARY)
    segmenter = segment.Segmenter(cropped_image, classifier)
    images = segmenter.segment()

    features = [classify.compute_features(image) for image in images]
    pdb.set_trace()
    predicted_characters = [chr(classifier.predict(feature)) for feature in features]
    predicted_name = ''.join(predicted_characters)
    print 'predicted name:', predicted_name

    # r = roster.Roster('roster-exams2')
    # closest_roster_name = r.get_closest_name(predicted_name)
    # print 'closest roster name:', closest_roster_name
    # print 'correct name:', correct_names[i]

