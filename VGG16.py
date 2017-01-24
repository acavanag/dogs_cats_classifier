import numpy as np
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda, Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

class Vgg16:

    def __init__(self, imageSize=(224, 224)):
        self.imageSize = imageSize
        self.model = self.__build(imageSize)

    def __build(self, imageSize):
        model = Sequential() # Linear stack of layers
        model.add(Lambda(self.__preprocess, input_shape=(3,)+imageSize))

        self.__addConvBlock(model, 2, 64)
        self.__addConvBlock(model, 2, 128)
        self.__addConvBlock(model, 3, 256)
        self.__addConvBlock(model, 3, 512)
        self.__addConvBlock(model, 3, 512)

        model.add(Flatten())
        self.__addFullyConnectedBlock(model)
        self.__addFullyConnectedBlock(model)
        model.add(Dense(1000, activation='softmax'))

        return model

    def loadWeights(self, weights):
        self.model.load_weights(weights)

    def __addConvBlock(self, model, layers, filters):
        for i in range(layers):
            model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(filters, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    def __addFullyConnectedBlock(self, model):
        model.add(Dense(4096, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

    @staticmethod
    def __preprocess(input):
        input = input - np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))
        return input[:, ::-1] # rgb->bgr
