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
    # classifier.cross_validate()
    classifier.tune_svm()



if __name__ == '__main__':
    main()