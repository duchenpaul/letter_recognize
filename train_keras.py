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

import purge_models
import train_prep

import config

import embedding_project


IMG_SIZE = config.IMG_SIZE
model_dir = config.MODEL_DIR
LOG_DIR = config.MODEL_LOG
num_classes = len(config.char_set)

epoch = 1000
batch_size = 80

# Learning rate
LR = 1e-3 / 2
tag = '[32-64]'
dropOutRate = .5

MODELNAME = 'letter_recognation'.format()
MODELNAME_FILE = MODELNAME + '.model'
MODELNAME_FULL_PATH = os.path.join(model_dir, MODELNAME_FILE)


def gen_meta(X_test_dataset, Y_test_dataset):
    from PIL import Image
    sample_size = 100
    # print(X_test_dataset.shape)
    X_test_dataset_sample = X_test_dataset[:sample_size*sample_size, :, :, :]
    Y_test_dataset_sample = Y_test_dataset[:sample_size*sample_size]
    # print(X_test_dataset_sample.shape)
    img_array = X_test_dataset_sample.reshape(sample_size, sample_size, IMG_SIZE, IMG_SIZE)
    img_array_flat = np.concatenate([np.concatenate([x for x in row], axis=1) for row in img_array])
    img = Image.fromarray(np.uint8(255 * (1. - img_array_flat)))
    log_dir = os.path.join(LOG_DIR, MODELNAME)
    os.mkdir(log_dir)
    img.save(os.path.join(log_dir, 'images.jpg'))
    np.savetxt(os.path.join(log_dir, 'metadata.tsv'), np.where(Y_test_dataset_sample)[1], fmt='%d')


def data_preprocess():
    train_data = np.load('train_data.npy', allow_pickle=True)
    X_dataset = train_data[:, 0]
    X_dataset = np.array([np.array(x) for x in X_dataset])
    X_dataset = X_dataset.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    Y_dataset = train_data[:, 1]
    Y_dataset = np.array([np.array(x) for x in Y_dataset])
    # print(X_dataset.shape)
    # print(Y_dataset.shape)
    return X_dataset, Y_dataset


def buildModel(shape):
    model = Sequential(name=MODELNAME)
    model.add(Conv2D(32, 5, 5, input_shape=(
        shape[1], shape[2], 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5)))

    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropOutRate))

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
        monitor="val_loss", patience=10, verbose=1, mode="auto")

    embedding_layer_names = set(layer.name
                            for layer in model.layers
                            if layer.name.startswith('dense_'))

    tbCallBack = TensorBoard(log_dir=os.path.join(LOG_DIR, MODELNAME),  # log 目录
                             histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
                             #                  batch_size=batch_size,     # 用多大量的数据计算直方图
                             write_graph=True,  # 是否存储网络结构图
                             write_grads=True,  # 是否可视化梯度直方图
                             write_images=True,  # 是否可视化参数
                             embeddings_freq=10,
                             embeddings_data = X_dataset[:400],
                             embeddings_layer_names=embedding_layer_names,
                             embeddings_metadata=None
                             )

    # tbCallBack = embedding_project.TensorResponseBoard(log_dir=os.path.join(LOG_DIR, MODELNAME),  # log 目录
    #                          histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
    #                          #                  batch_size=batch_size,     # 用多大量的数据计算直方图
    #                          write_graph=True,  # 是否存储网络结构图
    #                          write_grads=True,  # 是否可视化梯度直方图
    #                          write_images=True,  # 是否可视化参数
    #                          embeddings_freq=10,
    #                          embeddings_layer_names=['dense_1'],
    #                          embeddings_data = X_dataset[:100],
    #                          embeddings_metadata='metadata.tsv',
    #                          val_size=len(X_dataset[:100]), img_path='images.jpg', img_size=[28, 28]
    #                          )

    model.fit(X_dataset, Y_dataset, epochs=epoch, shuffle=True, batch_size=batch_size,
              validation_split=0.1, callbacks=[callback, tbCallBack])
    model.save(MODELNAME_FULL_PATH)


if __name__ == '__main__':
    purge_models.purge_models()
    X_dataset, Y_dataset = data_preprocess()
    gen_meta(X_dataset, Y_dataset)
    shape = X_dataset.shape
    model = buildModel(shape)
    train_model(model, X_dataset, Y_dataset)


