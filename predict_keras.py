import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os

import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model

import train_prep
import config
import color_card


model_dir = config.MODEL_DIR
IMG_SIZE = config.IMG_SIZE

MODELNAME = 'letter_recognation'
MODELNAME_FILE = MODELNAME + '.model'
MODELNAME_FULL_PATH = os.path.join(model_dir, MODELNAME_FILE)


def test_model():
    print('Test model')
    try:
        model = load_model(MODELNAME_FULL_PATH)
    except:
        print('Failed to load Model')
    else:
        print('Model loaded!')

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
        notSureFlag = ''
        notSureFlag = '?' if model_out[np.argmax(model_out)] < 0.5 else notSureFlag
        notSureFlag = '!' if model_out[np.argmax(model_out)] == 1 else notSureFlag
        confidence = model_out[np.argmax(model_out)]
        print('Answer: {}  Confidence: {}\n'.format(ans, confidence))
        y.imshow(orig, cmap='gray')
        color = color_card.color_card(confidence*100)
        plt.title(ans + notSureFlag,color=color)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)

    plt.show()


if __name__ == '__main__':
    test_model()