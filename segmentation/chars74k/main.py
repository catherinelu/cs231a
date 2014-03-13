import classify, roster, segment

classifier = classify.read_classifier()
segmenter = segment.Segmenter('adam.png')

images = segmenter.segment()


features = [classify.compute_features(image) for image in images]
predicted_characters = [chr(classifier.predict(feature)) for feature in features]
predicted_name = ''.join(predicted_characters)
print predicted_name

r = roster.Roster('roster-exams2')
closest_roster_name = r.get_closest_name(predicted_name)
print 'closest name:', closest_roster_name

