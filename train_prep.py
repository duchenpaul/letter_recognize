import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
# from tqdm import tqdm
import random

import config

TRAINDIR = config.TRAINING_DATA_DIR
TESTDIR = config.TEST_DATA_DIR


CATEGORIES = config.char_set
IMG_SIZE = config.IMG_SIZE


def number2list(number):
    n = len(CATEGORIES)
    listofzeros = [0] * n
    listofzeros[number] = 1
    return listofzeros


def create_training_data():
    training_data = []
    for idx, category in enumerate(CATEGORIES):  # do dogs and cats
        # create path to dogs and cats
        path = os.path.join(TRAINDIR, str(category))
        label = number2list(idx)
        for img in os.listdir(path):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(
                    path, img), cv2.IMREAD_GRAYSCALE)  # convert to array

                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                ret,new_array=cv2.threshold(new_array,100,255,cv2.THRESH_BINARY)
                training_data.append([new_array/255, label])
                # plt.imshow(new_array, cmap='gray')
                # plt.show()
            except Exception as e:
                print('Error Reading: ' + os.path.join(path, img))
                os.remove(os.path.join(path, img))
                pass
        #     break
        # break
    random.shuffle(training_data)
    # print(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def process_test_data():
    test_data = []
    path = TESTDIR  # create path to dogs and cats
    for img in os.listdir(path):  # iterate over each image per dogs and cats
        try:
            print(os.path.join(path, img))
            img_array = cv2.imread(os.path.join(
                path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
            # plt.imshow(img_array, cmap='gray')  # graph it
            # plt.show()  # display!

            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            # plt.imshow(new_array, cmap='gray')
            # plt.show()
            test_data.append([new_array, img.split('.')[0]])
        except Exception as e:
            print('Error Reading: ' + os.path.join(path, img))
            print(e)
            # os.remove(os.path.join(path, img))
            pass
        # break
    return test_data


if __name__ == '__main__':
    training_data = create_training_data()
