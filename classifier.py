from VGG16 import Vgg16
from keras.layers.core import Dense
from keras.preprocessing import image
from keras.optimizers import RMSprop

class Classifier:

    def __init__(self, output_class_count=None, image_size=(224, 224), weights=None):
        self.vgg16 = Vgg16(image_size)
        self.model = self.vgg16.model
        self.vgg16.loadWeights('./vgg16_bn.h5')
        self.__augment_model(self.model, output_class_count)

        if weights != None:
            self.vgg16.loadWeights(weights)


    def __augment_model(self, model, output_class_count):
        if output_class_count == None:
            return

        model.pop()
        for layer in model.layers:
            layer.trainable = False

        model.add(Dense(output_class_count, activation='softmax'))

    def __get_image_batches(self, path, gen=image.ImageDataGenerator(),
                          shuffle=True, image_size=(224, 224),
                          batch_size=8,
                          class_mode='categorical'):
        return gen.flow_from_directory(path,
                                       target_size=image_size,
                                       class_mode=class_mode,
                                       shuffle=shuffle,
                                       batch_size=batch_size)

    def __fit(self, model, batches, validation_batches, nb_epoch=1):
        model.fit_generator(batches,
                            samples_per_epoch=batches.N,
                            nb_epoch=nb_epoch,
                            validation_data=validation_batches,
                            nb_val_samples=validation_batches.N)

    def train(self, train_path, validation_path, output_path):
        train_batches = self.__get_image_batches(train_path)
        validation_batches = self.__get_image_batches(validation_path)

        optimizer = RMSprop(lr=0.1)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.__fit(self.model, train_batches, validation_batches)
        self.model.save_weights(output_path)

    def predict(self, input):
        predictions = self.model.predict(input)
        print(predictions)

    def summary(self):
        print(self.model.summary())