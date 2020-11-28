import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path
import os

from sklearn.model_selection import train_test_split
from sklearn import cluster
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

def read_images():
    images = []
    labels = []
    calssified_images = {}

    for class_id, class_dir in enumerate(sorted(Path('data').iterdir())):
        # nazwa folderu ze zdjeciami jako klucz slownika
        class_name = str(class_dir)
        class_name = class_name[5:]
        # Nazwy katalogu ze zdjeciami nazwami klas
        calssified_images.update({class_name: []})
        for image_path in class_dir.iterdir():
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            calssified_images[class_name].append(image)
            images.append(image)
            labels.append(class_name)
    # print(labels)
    return images, calssified_images, labels

def show_images(image_data):
    fix, ax = plt.subplots(nrows=5, ncols=5, figsize=(8, 16))

    for class_id, image in image_data.items():
        for row, img in enumerate(image[:5]):
            ax[row, class_id].imshow(img[..., ::-1])
            ax[row, class_id].axis('off')
    plt.show()


def divide_set(image_data, labels):
    train_images, valid_images, train_labels, valid_labels = train_test_split(image_data, labels, train_size=0.7, random_state=42, stratify=labels)
    print('Len train_images', len(train_images))
    print('Len train_labels', len(train_labels))
    print('Len valid_images', len(valid_images))
    print('Len valid_labels', len(valid_labels))

    return train_images, valid_images, train_labels, valid_labels

def convert_descriptors_to_histogram(descriptors, vocab_model, n_bins):
    assigned_words = vocab_model.predict(descriptors)
    # print(assigned_words)
    histogram, _ = np.histogram(assigned_words, bins=n_bins)
    histogram = histogram.astype(np.float32)/np.sum(histogram)
    # print(histogram)

    return histogram


def apply_feature_transform(images, feature_detector_descriptor, vocab_model, n_bins):
    histograms = []
    for image in images:
        _, descriptors =  feature_detector_descriptor.detectAndCompute(image, None)
        histogram = convert_descriptors_to_histogram(descriptors, vocab_model, n_bins)
        histograms.append(histogram)
    return histograms



def projekt():
    # TODO filtrowanie zlych obrazow
    # TODO zwiekszenie ilosci zbioru treningowego poprzez augmentacje
    images, classified_images, labels = read_images()
    print(classified_images.keys())
    print(len(classified_images[0]))
    print(len(images))

    # show_images(classified_images)
    train_images, valid_images, train_labels, valid_labels = divide_set(images, labels)
    # clusters_()

    feature_detector_descriptor = cv2.ORB_create()
    # print(feature_detector_descriptor)
    # print(type(feature_detector_descriptor))



    train_descriptors = []

    for image in train_images:
        for descriptor in feature_detector_descriptor.detectAndCompute(image, None)[1]:
            train_descriptors.append(descriptor)

    #
    #
    # print(len(train_descriptors))
    #
    #
    NB_WORDS = 128
    kmeans = cluster.KMeans(n_clusters=NB_WORDS, random_state=42)
    kmeans.fit(train_descriptors)
    print(kmeans.cluster_centers_.shape)


    descriptors = feature_detector_descriptor.detectAndCompute(valid_images[0], None)[1]
    # print(descriptors)
    # convert_descriptors_to_histogram(descriptors, kmeans, NB_WORDS)


    X_train = apply_feature_transform(train_images, feature_detector_descriptor, kmeans, NB_WORDS)
    y_train = train_labels
    X_valid =apply_feature_transform(valid_images, feature_detector_descriptor, kmeans, NB_WORDS)
    y_valid = valid_labels

    # classifier = DecisionTreeClassifier()
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

    pickle.dump(classifier, open('./clf.p', 'wb'))

    print('RandomForestClassifier:')
    print(classifier.score(X_train, y_train))
    print(classifier.score(X_valid, y_valid))

    param_grid = {
        'max_depth': [1, 5, 10, 30, 100],
        'n_estimators': [1, 5, 10, 50, 100],
        'criterion': ['gini', 'entropy']
    }

    k_fold = KFold(n_splits=5)

    grid_search = GridSearchCV(classifier, param_grid, cv=k_fold)
    grid_search.fit(X_train, y_train)

    print('grid serch:')
    print(grid_search.score(X_train, y_train))
    print(grid_search.score(X_valid, y_valid))

    print(grid_search.best_params_)




def testy():
    feature_detector_descriptor = cv2.ORB_create()
    feature_detector_descriptor = cv2.AKAZE_create()

    image = cv2.imread('doge.jpg', cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image is None:
      raise FileNotFoundError("Error loading image")


    fast  = cv2.FastFeatureDetector_create()
    orb = cv2.ORB_create()
    akaze = cv2.AKAZE_create()

    keypoints = fast.detect(image_grayscale)

    fast_image = cv2.drawKeypoints(image, keypoints, None, (255, 0 , 0))
    print('FAST', len(keypoints))

    keypoints = orb.detect(image_grayscale)
    orb_image = cv2.drawKeypoints(image, keypoints, None, (255, 0 , 0))
    print('ORB', len(keypoints))

    keypoints = akaze.detect(image_grayscale)
    keypoints, image_descriptors = feature_detector_descriptor.detectAndCompute(image, None)
    akaze_image = cv2.drawKeypoints(image, keypoints, None, (255, 0 , 0))
    print('AKAZE', len(keypoints), len(image_descriptors))
    print(keypoints)
    print(image_descriptors[0])


    fig, ax = plt.subplots(nrows=1  , ncols=3, figsize=(20,6))
    ax[0].imshow(fast_image)
    ax[1].imshow(orb_image)
    ax[2].imshow(akaze_image)
    plt.show()






def main():
    projekt()


if __name__ == '__main__':
    main()

