from classify import Classifier
from crop_name import CropName
import pdb
from segment import Segmenter
import sys
import cv2

ROSTER_FILENAME = 'roster-exams2'
IMAGE_DIRECTORY = 'exams2-test'

def main():
    # Get the lexicon
    lexicon = []
    lexicon.append('Alp Goel')  # TODO: Remove
    f = open(ROSTER_FILENAME)
    for line in f:
        lexicon.append(line.strip())
    f.close()

    # Create classifier
    classifier = Classifier(lexicon)

    # Load data and svm from pickled files
    # classifier.load_training_data()
    # classifier.load_svm()

    # Create pickled files
    classifier.train('kassel')
    classifier.train_svm()
    classifier.pickle_svm()
    classifier.pickle_training_data()

    crop = CropName(IMAGE_DIRECTORY)
    cropped_images, names = crop.run()

    num_correct = 0
    num_incorrect = 0
    for i, name in enumerate(names):
        # Classify a result
        print name
        cv2.imwrite('tmp.jpeg', cropped_images[i])

        segmenter = Segmenter('tmp.jpeg')  # TODO: Change to pixels
        characters = segmenter.segment()
        segmenter.show_image()

        predicted = classifier.classify(characters)
        print 'The classifier predicted:', predicted, 'from num characters:', len(characters)
        if name == predicted:
            num_correct += 1
        else:
            num_incorrect += 1

    print 'Num correct:', num_correct
    print 'Num num_incorrect', num_incorrect


if __name__ == '__main__':
    main()