import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn import cluster
from sklearn.svm import SVC



def read_images():
    images = []
    labels = []

    for class_id, class_dir in enumerate(sorted(Path('data').iterdir())):
        for image_path in class_dir.iterdir():
            # image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            # background = np.zeros((500, 750, 3), np.uint8)
            background = np.zeros((500, 750), np.uint8)
            image_width = float(np.shape(image)[1])
            image_height = float(np.shape(image)[0])
            background_width = float(np.shape(background)[1])
            background_height = float(np.shape(background)[0])


            if (image_width/background_width > 1) or (image_height/background_height > 1):
                # print(image_path)
                # print("width ratio:", image_width/background_width, 'height ratio:', image_height/background_height)
                width_ratio = image_width/background_width
                height_ratio = image_height/background_height
                # resize_img = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                if width_ratio > height_ratio:
                    resize_img = cv2.resize(image, None, fx=1/width_ratio, fy=1/width_ratio, interpolation=cv2.INTER_AREA)
                else:
                    resize_img = cv2.resize(image, None, fx=1/height_ratio, fy=1/height_ratio, interpolation=cv2.INTER_AREA)

                resize_img_width = np.shape(resize_img)[1]
                resize_img_height = np.shape(resize_img)[0]

                # przypisanie do tla obrazu zmniejszonego do przyjetych wymiarow
                background[0:int(resize_img_height), 0:int(resize_img_width)] = resize_img
                # zapisanie tla do zmiennej image
                image = background


                # cv2.imshow('img', image)
                # cv2.imshow('resize_img', resize_img)
                # cv2.waitKey()

            else:
                # przypisanie do tla obrazu mniejszego niz przyjete wymiary
                background[0:int(image_height), 0:int(image_width)] = image
                # zapisanie tla do zmiennej image
                image = background

            # cv2.imshow('img', image)
            # cv2.waitKey()

            images.append(image)
            labels.append(class_id)

    return images, labels

def show_images(image_data):
    fix, ax = plt.subplots(nrows=5, ncols=5, figsize=(8, 16))

    for class_id, image in image_data.items():
        for row, img in enumerate(image[:5]):
            ax[row, class_id].imshow(img[..., ::-1])
            ax[row, class_id].axis('off')
    plt.show()


def divide_set(image_data, labels):
    train_images, valid_images, train_labels, valid_labels = train_test_split(image_data, labels, train_size=0.7, random_state=42, stratify=labels)
    # print('Len train_images', len(train_images))
    # print('Len train_labels', len(train_labels))
    # print('Len valid_images', len(valid_images))
    # print('Len valid_labels', len(valid_labels))

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

    images, labels = read_images()
    print(labels)
    print(len(images))

    # show_images(classified_images)
    train_images, valid_images, train_labels, valid_labels = divide_set(images, labels)

    feature_detector_descriptor = cv2.AKAZE_create()

    train_descriptors = []

    for image in train_images:
        for descriptor in feature_detector_descriptor.detectAndCompute(image, None)[1]:
            train_descriptors.append(descriptor)


    print('train descriptors list finished')
    print('train descriptors len:', len(train_descriptors))

    NB_WORDS = 128
    kmeans = cluster.KMeans(n_clusters=NB_WORDS, random_state=42).fit(train_descriptors)

    print('kmeans fit finished')


    X_train = apply_feature_transform(train_images, feature_detector_descriptor, kmeans, NB_WORDS)
    y_train = train_labels
    X_valid =apply_feature_transform(valid_images, feature_detector_descriptor, kmeans, NB_WORDS)
    y_valid = valid_labels


    # Support Vector Classification
    classifier = SVC()
    classifier.fit(X_train, y_train)

    pickle.dump(kmeans, open('./vocab_model.p', 'wb'))
    pickle.dump(classifier, open('./clf.p', 'wb'))


    print('SVC:')
    print(classifier.score(X_train, y_train))
    print(classifier.score(X_valid, y_valid))



def main():
    projekt()


if __name__ == '__main__':
    main()

