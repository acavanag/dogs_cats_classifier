from classifier import Classifier


def train(ouput_class_count, image_size, train_path, validation_path, output_path):
    image_classifier = Classifier(ouput_class_count, image_size)
    image_classifier.summary()
    image_classifier.train(train_path, validation_path, output_path)


def predict(input, weights):
    image_classifier = Classifier(2, (224, 224), weights)
    image_classifier.summary()
    image_classifier.predict(input)

train(2,
      (224, 224),
      './dogscats/sample/train',
      './dogscats/sample/valid',
      './dogscats_weights')

# predict('', '')
