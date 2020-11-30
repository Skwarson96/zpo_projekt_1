import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path




def read_image():
    img_dict = {}
    for class_id, class_dir in enumerate(sorted(Path('data').iterdir())):
        class_name = str(class_dir)[5:]
        img_dict.update({class_name: []})
        for image_path in class_dir.iterdir():
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            img_dict[class_name].append(image)
            # cv2.imshow("img", image)
            # cv2.waitKey()

    return img_dict



def main():
    image_dict = read_image()
    print(image_dict)



if __name__ == '__main__':
    main()

