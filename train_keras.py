import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import os
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard

# import purge_models
import train_prep

import config


IMG_SIZE = config.IMG_SIZE
model_dir = config.MODEL_DIR
log_dir = config.MODEL_LOG
num_classes = len(config.char_set)

epoch = 1500
batch_size = 80

# Learning rate
LR = 1e-3 / 2
tag = '[32-64]'
dropOutRate = .5

MODELNAME = 'letter_recognation-{}-{}.model'.format(
    LR, '{}_e{}'.format(tag, epoch))
MODELNAME_FILES = os.path.join(model_dir, MODELNAME)


def data_preprocess():
    train_data = np.load('train_data.npy', allow_pickle=True)
    X_dataset = train_data[:, 0]
    X_dataset = np.array([np.array(x) for x in X_dataset])
    X_dataset = X_dataset.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    Y_dataset = train_data[:, 1]
    Y_dataset = np.array([np.array(x) for x in Y_dataset])
    print(X_dataset.shape)
    print(Y_dataset.shape)
    return X_dataset, Y_dataset


def buildModel(shape):
    model = Sequential()
    model.add(Conv2D(32, 5, 5, input_shape=(
        shape[1], shape[2], 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5)))

    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropOutRate))

    model.add(Dense(num_classes, activation='softmax'))

    opt = Adam(lr=LR)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model


def train_model(model, X_dataset, Y_dataset):
    print("TRAIN")
    callback = EarlyStopping(
        monitor="loss", patience=30, verbose=1, mode="auto")
    tbCallBack = TensorBoard(log_dir=log_dir,  # log 目录
                             histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
                             #                  batch_size=batch_size,     # 用多大量的数据计算直方图
                             write_graph=True,  # 是否存储网络结构图
                             write_grads=True,  # 是否可视化梯度直方图
                             write_images=True,  # 是否可视化参数
                             embeddings_freq=0,
                             embeddings_layer_names=None,
                             embeddings_metadata=None)

    model.fit(X_dataset, Y_dataset, epochs=epoch, shuffle=True, batch_size=batch_size,
              validation_split=0.2, callbacks=[callback, tbCallBack])
    model.save(MODELNAME_FILES)


if __name__ == '__main__':
    # purge_models.purge_models()
    X_dataset, Y_dataset = data_preprocess()
    shape = X_dataset.shape
    model = buildModel(shape)
    train_model(model, X_dataset, Y_dataset)

