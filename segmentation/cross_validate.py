import classify, pickle, pdb
import random
from sklearn import neighbors


def cross_validate(training_images, training_labels, n=10):
    num_training_images = len(training_images)

    # Shuffle the training images and labels
    shuffled_index = range(num_training_images)
    random.shuffle(shuffled_index)
    shuffled_training_images = []
    shuffled_training_labels = []
    for i in shuffled_index:
        shuffled_training_images.append(training_images[i])
        shuffled_training_labels.append(training_labels[i])
    training_images = shuffled_training_images
    training_labels = shuffled_training_labels


    n_training_image_groups = [None for i in xrange(n)]
    n_training_label_groups = [None for i in xrange(n)]
    for i in xrange(n):
        start_index = int(float(i) / n * num_training_images)
        end_index = int(float(i + 1) / n * num_training_images)
        n_training_image_groups[i] = training_images[start_index:end_index]
        n_training_label_groups[i] = training_labels[start_index:end_index]


    num_total_correct = 0
    num_total_incorrect = 0
    for i in xrange(n):
        print 'Testing fold %d...' % i

        num_correct = 0
        num_incorrect = 0

        train_images = []
        train_labels = []
        for j in xrange(n):
            if j != i:
                train_images += n_training_image_groups[j]
                train_labels += n_training_label_groups[j]
        classifier = neighbors.KNeighborsClassifier(10)
        classifier.fit(train_images, train_labels)

        # # Save the classifier to a file
        # original_filename = classify.CLASSIFIER_FILE_NAME
        # classify.CLASSIFIER_FILE_NAME += '-%d-fold' % i
        # classify.store_classifier(classifier)
        # classify.CLASSIFIER_FILE_NAME = original_filename


        test_images = n_training_image_groups[i]
        test_labels = n_training_label_groups[i]
        predicted_labels = classifier.predict(test_images)

        for j, predicted_label in enumerate(predicted_labels):
            if predicted_label == test_labels[j]:
                num_correct += 1
            else:
                num_incorrect += 1

        print '%d correct and %d incorrect (%d%% accuracy)' % (num_correct,
            num_incorrect, int(num_correct / float(num_correct + num_incorrect) * 100))

        num_total_correct += num_correct
        num_total_incorrect += num_incorrect

    print 'Final stats:'
    print '%d total correct and %d total incorrect (%d%% accuracy)' % (num_total_correct,
        num_total_incorrect, int(num_total_correct / float(num_total_correct + num_total_incorrect) * 100))


if __name__ == '__main__':
    # images, gray_images = classify.read_training_images()
    # training_images = []
    # training_labels = []
    # for letter in gray_images:
    #     for image in gray_images[letter]:
    #         training_images.append(classify.compute_features(image))
    #         training_labels.append(ord(letter))

    # features_file = open('training_image_features', 'w')
    # pickle.dump(training_images, features_file)
    # labels_file = open('training_labels', 'w')
    # pickle.dump(training_labels, labels_file)

    training_images = pickle.load(open('training_image_features'))
    training_labels = pickle.load(open('training_labels'))

    # classifier = neighbors.KNeighborsClassifier(10)
    # classifier.fit(training_images, training_labels)

    # c = open('classifier-knn', 'w')
    # pickle.dump(classifier, c)
    # pdb.set_trace()


    cross_validate(training_images, training_labels)

