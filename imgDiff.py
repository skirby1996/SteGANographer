
import sys
import os

import cv2
import numpy as np

import datetime


def split_into_rgb_channels(image):
    '''Split the target image into its red, green and blue channels.
    image - a numpy array of shape (rows, columns, 3).
    output - three numpy arrays of shape (rows, columns) and dtype same as
             image, containing the corresponding channels.
    '''
    red = image[:, :, 2]
    green = image[:, :, 1]
    blue = image[:, :, 0]
    return red, green, blue


def main():
    '''temporary --- path for local testing'''
    root_dir = os.path.abspath("")
    img_dir = os.path.join(root_dir, "Images")

    '''create directory --- named by date and time --- to store the output images'''
    date_time = datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")
    out_dir = os.path.join(img_dir, date_time)
    os.mkdir(out_dir)

    '''get original image that has been inputted in the Neural Network'''
    user_in = input("Please enter name of original image: ")
    original_img_path = os.path.join(img_dir, user_in)

    '''get image that has been outputted by the Neural Network'''
    user_in = input("Please enter name of outputted image: ")
    outputted_img_path = os.path.join(img_dir, user_in)

    '''check that images exist'''
    if os.path.exists(original_img_path) and os.path.exists(outputted_img_path):
        original_img = cv2.imread(original_img_path, cv2.IMREAD_COLOR)
        outputted_img = cv2.imread(outputted_img_path, cv2.IMREAD_COLOR)
    else:
        if not os.path.exists(original_img_path):
            print("Error: No image at %s" % original_img_path)
        if not os.path.exists(outputted_img_path):
            print("Error: No image at %s" % outputted_img_path)

    difference_img = np.absolute(original_img - outputted_img)
    cv2.imwrite(os.path.join(out_dir, "difference.png"), difference_img)

    '''split difference image into red green and blue color channels'''
    red, green, blue = split_into_rgb_channels(difference_img)
    for values, color, channel in zip((red, green, blue), ('red', 'green', 'blue'), (2, 1, 0)):
        difference_img = np.zeros(
            (values.shape[0], values.shape[1], 3), dtype=values.dtype)
        difference_img[:, :, channel] = values
        print("Saving Image: %s." % color + '.png')
        cv2.imwrite(os.path.join(out_dir, color+"_out.png"), difference_img)


if __name__ == "__main__":
    main()
