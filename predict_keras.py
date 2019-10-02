import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os

import numpy as np
from keras.models import load_model

import train_prep
import config


model_dir = config.MODEL_DIR
IMG_SIZE = config.IMG_SIZE

MODELNAME = ''
MODELNAME_FILES = os.path.join(model_dir, MODELNAME)


def test_model():
    print('Test model')
    if os.path.exists(os.path.join(model_dir, '{}.meta'.format(MODELNAME))):
        model = load_model(MODELNAME_FILES)
        print('Model loaded!')

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
        print('Answer: {}  Confidence: {}\n'.format(
            ans, model_out[np.argmax(model_out)]))
        y.imshow(orig, cmap='gray')
        plt.title(ans + notSureFlag)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()


if __name__ == '__main__':
    test_model()