import os
import numpy as np
from datetime import datetime
# import tflearn
# from tflearn.layers.conv import conv_2d, max_pool_2d
# from tflearn.layers.core import input_data, dropout, fully_connected
# from tflearn.layers.estimator import regression

import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPool2D

import purge_models
import train_prep

import config

import color_card


IMG_SIZE = config.IMG_SIZE
model_dir = config.MODEL_DIR

epoch = 10

# Learning rate
LR = 1e-3/2
layer_list = [16, 32]
tag = '[{}]'.format('-'.join([str(x) for x in layer_list]))

MODELNAME = os.path.join('letter_recognation-{}-{}.model'.format(LR, '{}_e{}'.format(tag, epoch)))
MODELNAME_FILE = os.path.join(model_dir, MODELNAME)

logdir = "log/" + datetime.now().strftime("%Y%m%d-%H%M%S")
with tf.compat.v1.Session() as sess:
    file_writer = tf.compat.v1.summary.FileWriter(logdir, sess.graph)
# file_writer.set_as_default()
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


model = tf.keras.Sequential()
model.add(Conv2D(layer_list[0], (5, 5), padding="same", input_shape=(IMG_SIZE, IMG_SIZE, 1), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.8))

model.add(Conv2D(layer_list[1], (5, 5), padding="same", activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.8))

# Hidden layer with 500 nodes
model.add(Flatten(input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(Dense(500, activation="relu"))

# Output layer with 32 nodes (one for each possible letter/number we predict)
model.add(Dense(len(config.char_set), activation="softmax"))

# Ask Keras to build the TensorFlow model behind the scenes
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])



def train_model():
    purge_models.purge_models()
    train_data = np.load('train_data.npy', allow_pickle=True)
    # train_data = train_prep.create_training_data()
    # train = train_data[:-5]
    train = train_data[:-500]
    test = train_data[-500:]

    X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    # X_train = tf.keras.utils.normalize([i[0] for i in train], axis=1)
    Y_train = np.array([i[1] for i in train])
    # print(X_train[0])

    X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    # X_test = tf.keras.utils.normalize([i[0] for i in test], axis=1)
    Y_test = np.array([i[1] for i in test])

    print("TRAIN")
    # Train the neural network
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=epoch, verbose=3, callbacks=[tensorboard_callback],)

    # Save the trained model to disk
    model.save(MODELNAME_FILE)


def test_model():
    print('Test model')
    # if os.path.exists(os.path.join(model_dir, '{}.meta'.format(MODELNAME))):
    #     model.load(MODELNAME_FILE)
    #     print('Model loaded!')
    # else:
    #     err_msg = 'Model {} not found'.format(os.path.join(model_dir, '{}.meta'.format(MODELNAME)))
    #     raise Exception(err_msg)


    import matplotlib.pyplot as plt

    # if you need to create the data:
    test_data = train_prep.process_test_data()
    # if you already have some saved:
    # test_data = np.load('test_data.npy')
    print('processed test data {}'.format(len(test_data)))
    fig = plt.figure()

    for num, data in enumerate(test_data[:12]):
        img_num = data[1]
        img_data = data[0]

        y = fig.add_subplot(3, 4, num + 1)
        orig = img_data
        data = img_data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        #model_out = model.predict([data])[0]
        model_out = model.predict([data])[0]

        # print(model_out)
        str_label = str(np.argmax(model_out))
        # print(str_label)
        # print(config.char_set[int(np.argmax(model_out))])
        ans = chr(config.char_set[int(np.argmax(model_out))])
        notSureFlag = '?' if model_out[np.argmax(model_out)] < 0.5 else ''
        confidence = model_out[np.argmax(model_out)]
        print('Answer: {}  Confidence: {}\n'.format(ans, confidence))
        y.imshow(orig, cmap='gray')
        color = color_card.color_card(confidence*100)
        plt.title(ans + notSureFlag,color=color)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()


if __name__ == '__main__':
    train_model()
    test_model()
