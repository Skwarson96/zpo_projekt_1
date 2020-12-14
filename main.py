import json
import os
import pickle
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


# TODO: versions of libraries that will be used:
#  Python 3.9 (you can use previous versions as well)
#  numpy 1.19.4
#  scikit-learn 0.22.2.post1
#  opencv-python 4.2.0.34


def load_dataset(dataset_dir_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i, class_dir in enumerate(sorted(dataset_dir_path.iterdir())):
        # print(i, class_dir)
        for file in class_dir.iterdir():
            img_file = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            # img_file = cv2.imread(str(file), cv2.IMREAD_COLOR)
            x.append(img_file)
            y.append(i)

    return np.asarray(x), np.asarray(y)


def convert_descriptor_to_histogram(descriptors, vocab_model, normalize=True) -> np.ndarray:
    features_words = vocab_model.predict(descriptors)
    histogram = np.zeros(vocab_model.n_clusters, dtype=np.float32)
    unique, counts = np.unique(features_words, return_counts=True)
    histogram[unique] += counts
    if normalize:
        histogram /= histogram.sum()

    return histogram


def apply_feature_transform(data: np.ndarray, feature_detector_descriptor, vocab_model) -> np.ndarray:
    data_transformed = []
    for image in data:
        keypoints, image_descriptors = feature_detector_descriptor.detectAndCompute(image, None)
        bow_features_histogram = convert_descriptor_to_histogram(image_descriptors, vocab_model)
        data_transformed.append(bow_features_histogram)

    return np.asarray(data_transformed)


def data_processing(x: np.ndarray) -> np.ndarray:
    # TODO: add data processing here

    for idx in range(len(x)):
        image = x[idx]
        # background = np.zeros((500, 750, 3), np.uint8)
        background = np.zeros((500, 750), np.uint8)
        image_width = float(np.shape(image)[1])
        image_height = float(np.shape(image)[0])
        background_width = float(np.shape(background)[1])
        background_height = float(np.shape(background)[0])

        if (image_width / background_width > 1) or (image_height / background_height > 1):

            width_ratio = image_width / background_width
            height_ratio = image_height / background_height

            if width_ratio > height_ratio:
                resize_img = cv2.resize(image, None, fx=1 / width_ratio, fy=1 / width_ratio,
                                        interpolation=cv2.INTER_AREA)
            else:
                resize_img = cv2.resize(image, None, fx=1 / height_ratio, fy=1 / height_ratio,
                                        interpolation=cv2.INTER_AREA)

            resize_img_width = np.shape(resize_img)[1]
            resize_img_height = np.shape(resize_img)[0]

            # przypisanie do tla obrazu zmniejszonego do przyjetych wymiarow
            background[0:int(resize_img_height), 0:int(resize_img_width)] = resize_img
            # zapisanie tla do zmiennej image
            image = background

        else:
            # przypisanie do tla obrazu mniejszego niz przyjete wymiary
            background[0:int(image_height), 0:int(image_width)] = image
            # zapisanie tla do zmiennej image
            image = background
        x[idx] = image


    return x


def project():
    np.random.seed(42)

    # TODO: fill the following values
    first_name = 'Maciej'
    last_name = 'Skwara'

    data_path = Path('data_testowe')  # You can change the path here
    data_path = os.getenv('DATA_PATH', data_path)  # Don't change that line
    x, y = load_dataset(data_path)
    x = data_processing(x)

    # TODO: create a detector/descriptor here. Eg. cv2.AKAZE_create()
    feature_detector_descriptor = cv2.AKAZE_create()

    # TODO: train a vocabulary model and save it using pickle.dump function
    with Path('vocab_model.p').open('rb') as vocab_file:  # Don't change the path here
        vocab_model = pickle.load(vocab_file)

    x_transformed = apply_feature_transform(x, feature_detector_descriptor, vocab_model)

    # TODO: train a classifier and save it using pickle.dump function
    with Path('clf.p').open('rb') as classifier_file:  # Don't change the path here
        clf = pickle.load(classifier_file)

    score = clf.score(x_transformed, y)
    print(f'{first_name} {last_name} score: {score}')
    with Path(f'{last_name}_{first_name}_score.json').open('w') as score_file:  # Don't change the path here
        json.dump({'score': score}, score_file)


if __name__ == '__main__':
    project()
