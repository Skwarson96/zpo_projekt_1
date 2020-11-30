import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random
import sys, os


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
            os.rmdir(path)
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


def main():
    image_dict, image_names = read_image()
    rotation(image_dict, image_names)



    # print(image_dict)



if __name__ == '__main__':
    main()

