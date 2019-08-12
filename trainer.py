import os
import numpy as np

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import purge_models
import train_prep

import config

import color_card

tflearn.init_graph(gpu_memory_fraction=0.5)

IMG_SIZE = config.IMG_SIZE
model_dir = config.MODEL_DIR

epoch = 1500

# Learning rate
LR = 1e-3/2
layer_list = [16, 32]
tag = '[{}]'.format('-'.join([str(x) for x in layer_list]))

MODELNAME = os.path.join('letter_recognation-{}-{}.model'.format(LR, '{}_e{}'.format(tag, epoch)))
MODELNAME_FILES = os.path.join(model_dir, MODELNAME)


convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, layer_list[0], 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = dropout(convnet, 0.8)

convnet = conv_2d(convnet, layer_list[1], 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = dropout(convnet, 0.8)


convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, len(config.char_set), activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR,
                     loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir=config.MODEL_LOG, tensorboard_verbose=3)


def train_model():
    purge_models.purge_models()
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

    print("TRAIN")
    model.fit({'input': X}, {'targets': Y}, n_epoch=epoch, validation_set=({'input': test_x}, {'targets': test_y}),
              snapshot_step=500, show_metric=True, run_id=MODELNAME)
    model.save(MODELNAME_FILES)


def test_model():
    print('Test model')
    if os.path.exists(os.path.join(model_dir, '{}.meta'.format(MODELNAME))):
        model.load(MODELNAME_FILES)
        print('Model loaded!')
    else:
        err_msg = 'Model {} not found'.format(os.path.join(model_dir, '{}.meta'.format(MODELNAME)))
        raise Exception(err_msg)


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
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
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
    # train_model()
    test_model()
