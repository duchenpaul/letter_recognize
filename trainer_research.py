import os
import numpy as np

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf

import train_prep

import config

tflearn.init_graph(gpu_memory_fraction=0.5)

IMG_SIZE = config.IMG_SIZE
model_dir = config.MODEL_DIR

epoch = 15

# Learning rate
LR = 1e-3

model_plan_list = [[32, 64], [8, 16], [16, 32], [64, 128], [32], [64], [128], ]
    


def create_model(nb_filter_list):
    tag = '[{}]'.format('-'.join([str(x) for x in nb_filter_list]))
    MODELNAME = os.path.join(model_dir, 'letter_recognation-{}-{}.model'.format(LR, '{}_e{}'.format(tag, epoch)))

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
    for nb_filter in nb_filter_list:
        convnet = conv_2d(convnet, nb_filter, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, len(config.char_set), activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR,
                         loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')
    return MODELNAME, model


def train_model():
    train_data = np.load('train_data.npy', allow_pickle=True)
    # train_data = train_prep.create_training_data()
    # train = train_data[:-5]
    train = train_data[:-500]
    test = train_data[-500:]

    X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    Y = [i[1] for i in train]
    # print(Y)

    test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    test_y = [i[1] for i in test]

    for nb_filter_list in model_plan_list:
        print("TRAIN {}".format(str(nb_filter_list)))
        MODELNAME, model = create_model(nb_filter_list)
        model.fit({'input': X}, {'targets': Y}, n_epoch=epoch, validation_set=({'input': test_x}, {'targets': test_y}),
                  snapshot_step=500, show_metric=True, run_id=MODELNAME)
        model.save(MODELNAME)
        tf.reset_default_graph()


if __name__ == '__main__':
    train_model()
