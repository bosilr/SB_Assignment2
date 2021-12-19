import os
import cv2
import numpy as np


def change_image_chunks(from_index, to_index, from_file, to_file):
    read_path = "C:/Users/Uporabnik/ROK_MAG_2.letnik/SB/Assignment2/data/ears/" + from_file
    write_path = "C:/Users/Uporabnik/ROK_MAG_2.letnik/SB/Assignment2/data/ears/processed/" + to_file

    for name in range(from_index, to_index):
        name_string = "000" + str(name)
        name_string = name_string[-4:]
        image_source = read_path + "/" + name_string + ".png"
        img = cv2.imread(image_source)

        # greyscale
        # intensity_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        # intensity_img[:, :, 0] = cv2.equalizeHist(intensity_img[:, :, 0])
        # img = cv2.cvtColor(intensity_img, cv2.COLOR_YCrCb2BGR)

        # img = cv2.equalizeHist(img)

        ####################################################
        # image sharpening

        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

        ####################################################
        # edge detection

        # img = cv2.Canny(img, 100, 200)


        ######################################################
        image_dest = write_path + "/" + name_string + ".png"
        cv2.imwrite(image_dest, img)


def change_image():
    change_image_chunks(1, 251, "true_test", "sharpened/images/test")
    change_image_chunks(1, 651, "true_train", "sharpened/images/train")
    change_image_chunks(651, 751, "true_validation", "sharpened/images/val")


if __name__ == '__main__':
    print(os.getcwd())
    change_image()
