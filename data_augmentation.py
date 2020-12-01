import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random
import sys, os
import shutil

def read_image():
    img_dict = {}
    img_names = {}
    for class_id, class_dir in enumerate(sorted(Path('data').iterdir())):
        class_name = str(class_dir)[5:]
        img_dict.update({class_name: []})
        img_names.update({class_name: []})
        for image_path in class_dir.iterdir():
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            img_dict[class_name].append(image)
            # take image name from image path
            img_name = str(image_path)[len(str(class_dir))+1:]
            img_names[class_name].append(img_name)
            # cv2.imshow("img", image)
            # cv2.waitKey()

    return img_dict, img_names

def rotation(img_dict, img_names):
    for key in img_dict:
        # reading path to file
        pathname = os.path.dirname(sys.argv[0])
        # make director for rotated images
        path = pathname + '/' + key + '_rotated_image'
        # check if the dir exists
        isFile = os.path.isdir(path)
        if isFile:
            # delete old dir
            shutil.rmtree(path, ignore_errors=True)
        # make new dir
        os.mkdir(path)

        for idx in range(len(img_dict[key])):
            image = img_dict[key][idx]

            # get a random angle to rotate image
            angle = random.randint(-15, 15)
            if angle == 0:
                angle = 5

            # get image center
            image_center = (np.shape(image)[0]/2, np.shape(image)[1]/2)
            # calculate rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            # warp affine
            rotate_img = cv2.warpAffine(image, rotation_matrix, (np.shape(image)[0], np.shape(image)[1]), flags=cv2.INTER_LINEAR)

            # cv2.imshow("rotate_img", rotate_img)
            # cv2.waitKey()

            # take image name from dict
            img_name = img_names[key][idx]
            # save rotated image
            cv2.imwrite(path + '/' + str(img_name) + '_rotated_' + str(angle), rotate_img)

def flip(img_dict, img_names):
    for key in img_dict:
        # reading path to file
        pathname = os.path.dirname(sys.argv[0])
        # make director for flipped images
        path = pathname + '/' + key + '_flipped_image'

        # check if the dir exists
        isFile = os.path.isdir(path)
        if isFile:
            # delete old dir
            shutil.rmtree(path, ignore_errors=True)
        # make new dir
        os.mkdir(path)

        for idx in range(len(img_dict[key])):
            image = img_dict[key][idx]
            # horizontal flip
            flipped_image = cv2.flip(image, 1)

            # cv2.imshow("flipped_image", flipped_image)
            # cv2.waitKey()

            # take image name from dict
            img_name = img_names[key][idx]
            # save flipped image
            cv2.imwrite(path + '/' + str(img_name) + '_flipped', flipped_image)

def lightening(img_dict, img_names):
    for key in img_dict:
        # reading path to file
        pathname = os.path.dirname(sys.argv[0])
        # make director for brightened images
        path = pathname + '/' + key + '_brightened_image'

        # check if the dir exists
        isFile = os.path.isdir(path)
        if isFile:
            # delete old dir
            shutil.rmtree(path, ignore_errors=True)
        # make new dir
        os.mkdir(path)

        for idx in range(len(img_dict[key])):
            image = img_dict[key][idx]

            brightened_image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            bright_value = random.randint(50, 100)

            brightened_image_hsv[:,:,2] = cv2.add(brightened_image_hsv[:,:,2], bright_value)
            brightened_image = cv2.cvtColor(brightened_image_hsv, cv2.COLOR_HSV2BGR)

            # cv2.imshow('image', image)
            # cv2.imshow('brightened_image', brightened_image)
            # cv2.waitKey()

            # take image name from dict
            img_name = img_names[key][idx]
            # save brightened image
            cv2.imwrite(path + '/' + str(img_name) + '_brightened', brightened_image)

def darkening(img_dict, img_names):
    for key in img_dict:
        # reading path to file
        pathname = os.path.dirname(sys.argv[0])
        # make director for brightened images
        path = pathname + '/' + key + '_darkened_image'

        # check if the dir exists
        isFile = os.path.isdir(path)
        if isFile:
            # delete old dir
            shutil.rmtree(path, ignore_errors=True)
        # make new dir
        os.mkdir(path)

        for idx in range(len(img_dict[key])):
            image = img_dict[key][idx]

            darkened_image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            dark_value = random.randint(-100, -50)

            darkened_image_hsv[:,:,2] = cv2.add(darkened_image_hsv[:,:,2], dark_value)
            darkened_image = cv2.cvtColor(darkened_image_hsv, cv2.COLOR_HSV2BGR)

            # cv2.imshow('image', image)
            # cv2.imshow('darkened_image', darkened_image)
            # cv2.waitKey()

            # take image name from dict
            img_name = img_names[key][idx]
            # save brightened image
            cv2.imwrite(path + '/' + str(img_name) + '_darkened', darkened_image)


def main():
    image_dict, image_names = read_image()
    # rotation(image_dict, image_names)
    # flip(image_dict, image_names)
    lightening(image_dict, image_names)
    darkening(image_dict, image_names)


if __name__ == '__main__':
    main()

