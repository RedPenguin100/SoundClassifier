import numpy as np
import sys
import os
import logging

import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers
import numpy as np
import sys
import os
import logging
from retrain_imagenet_classifier_tensorflow_2.sound_to_image import SPECTROGRAM_PATH, URBAN_SOUND8K_CSV_PATH, AUDIO_PATH

from retrain_imagenet_classifier_tensorflow_2.configuration import Configuration
from retrain_imagenet_classifier_tensorflow_2.sound_to_image import DEFAULT_SPECTROGRAM_PATH, URBAN_SOUND8K_CSV_PATH, AUDIO_PATH

tf.get_logger().setLevel('ERROR')
logging.getLogger('PIL').setLevel(logging.INFO)
logging.basicConfig(level=logging.DEBUG, filename='log.txt')

DATASET_SIZE = 8732
IMAGE_SHAPE = (224, 224)
CLASSES = ['air_conditioner', 'car_horn', 'children_playing',
           'dog_bark', 'drilling', 'engine_idling',
           'gun_shot', 'jackhammer', 'siren', 'street_music']
NUMBER_OF_CLASSES = len(CLASSES)

EXPORT_PATH = 'C:\\tmp\\saved_models\\spectogram'
FEATURE_EXTRACTOR_URL = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2'


def create_model(num_classes, verbose=False):
    # Now we want to retrain:
    feature_extractor_layer = hub.KerasLayer(FEATURE_EXTRACTOR_URL,
                                             input_shape=(224, 224, 3))

    feature_extractor_layer.trainable = False

    # Add the final layer to the ready model to train.
    model = tf.keras.Sequential([feature_extractor_layer, layers.Dense(num_classes,
                                                                       activation='softmax')])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['acc'])
    if verbose:
        print(model.summary())
    return model


def fix_gpu():
    """
    I don't fully understand this yet but these lines fix an error in CUDNN
    """

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


def wavfile_to_spectrogram_path(wavfile):
    return os.path.join(os.path.abspath(DEFAULT_SPECTROGRAM_PATH), wavfile) + '.png'


def get_image_train_data(split, batch_size=32):
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
    df = pd.read_csv(URBAN_SOUND8K_CSV_PATH)
    df = df[df['fold'] != split]
    df.loc[:, ('slice_file_name')] = df.loc[:, ('slice_file_name')].apply(wavfile_to_spectrogram_path)
    image_data = image_generator.flow_from_dataframe(df, directory=os.path.abspath(DEFAULT_SPECTROGRAM_PATH),
                                                     x_col='slice_file_name',
                                                     y_col='class',
                                                     target_size=IMAGE_SHAPE, batch_size=configuration.batch_size,
                                                     follow_links=True)
    return image_data


def get_image_test_data(spectrogram_path, fold, batch_size=32):
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
    df = pd.read_csv(URBAN_SOUND8K_CSV_PATH)
    df = df[df['fold'] == fold].copy()
    df.loc[:, ('slice_file_name')] = df.loc[:, ('slice_file_name')].apply(wavfile_to_spectrogram_path)
    image_data = image_generator.flow_from_dataframe(df, directory=os.path.abspath(DEFAULT_SPECTROGRAM_PATH),
                                                     x_col='slice_file_name',
                                                     y_col='class',
                                                     shuffle=False,
                                                     target_size=IMAGE_SHAPE, batch_size=batch_size,
                                                     target_size=IMAGE_SHAPE, batch_size=configuration.batch_size,
                                                     follow_links=True)
    return image_data


def get_all_data(batch_size=32):
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
    df = pd.read_csv(URBAN_SOUND8K_CSV_PATH)
    df.loc[:, ('slice_file_name')] = df.loc[:, ('slice_file_name')].apply(wavfile_to_spectrogram_path)
    image_data = image_generator.flow_from_dataframe(df, directory=os.path.abspath(SPECTROGRAM_PATH),
                                                     x_col='slice_file_name',
                                                     y_col='class',
                                                     target_size=IMAGE_SHAPE, batch_size=batch_size,
                                                     follow_links=True)
    return image_data


class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        self.model.reset_metrics()


def get_nth_image(image_data, n):
    batch_num = n // image_data.batch_size
    within_batch_index = n % image_data.batch_size
    return image_data[batch_num][0][within_batch_index]


def get_nth_label(image_data, n):
    batch_num = n // image_data.batch_size
    within_batch_index = n % image_data.batch_size
    return np.argmax(image_data[batch_num][1][within_batch_index])


def print_confusion_matrix(model, image_data):
    predicted_labels = np.argmax(model.predict_generator(image_data), axis=1)
    print(confusion_matrix(image_data.classes, predicted_labels))


def plot_next_n_errors(model, image_data, n):
    class_names = sorted(image_data.class_indices.items(), key=lambda pair: pair[1])
    class_names = np.array([key.title() for key, value in class_names])

    print(class_names)
    predicted_labels = model.predict_generator(image_data)
    predicted_labels_id = np.argmax(predicted_labels, axis=-1)
    predicted_labels_pictures_names = class_names[predicted_labels_id]
    mislabeled_indices = np.where(predicted_labels_id != image_data.classes)[0]
    plt.figure(figsize=(10, 9))
    plt.subplots_adjust(hspace=0.5)

    for n in range(min(30, len(mislabeled_indices))):
        plt.subplot(6, 5, n + 1)
        plt.imshow(get_nth_image(image_data, mislabeled_indices[n]))
        # color = 'green' if predicted_labels_id[n] == wanted_id else 'red'
        color = 'red'
        plt.title('predicted: ' + predicted_labels_pictures_names[mislabeled_indices[n]].title() + '\n actual: '
                  + class_names[get_nth_label(image_data, mislabeled_indices[n])], color=color)
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError('Usage: {} load/train/10fold should_create_hierarchy'.format(sys.argv[0]))
    configuration = Configuration()
    first_argument = sys.argv[1]
    should_load_10fold = should_load = should_train = should_10fold = False
    if first_argument == 'load':
        should_load = True
    elif first_argument == 'train':
        should_train = True
    else:
        raise ValueError('Bad parameter given')

    fix_gpu()

    # Obtain data to memory.
    # if should_create_hierarchy:
    #     try:
    #         create_sym_hierarchy(SPECTROGRAM_PATH)
    #     except FileExistsError:
    #         raise RuntimeError('FileExistsError was raised while creating hierarchy.' + os.linesep +
    #                            'Please delete the hierarchy and try again.')
    #
    if should_train:
        image_data = get_image_train_data(DEFAULT_SPECTROGRAM_PATH)
        model = create_model(num_classes=NUMBER_OF_CLASSES)
        steps_per_epoch = np.ceil(image_data.samples / configuration.batch_size)

    if should_train:
        for split_num in range(1, 11):
            image_data = get_image_train_data(DEFAULT_SPECTROGRAM_PATH, split=split_num)
            model = create_model(num_classes=NUMBER_OF_CLASSES)
            steps_per_epoch = np.ceil(image_data.samples / configuration.batch_size)

            batch_stats_callback = CollectBatchStats()

            model.fit_generator(image_data, epochs=2,
                                steps_per_epoch=steps_per_epoch,
                                callbacks=[batch_stats_callback])
            model.save(EXPORT_PATH + str(split_num) + 'split')
            # result = model.evaluate()
            # logging.debug('The result of split: {split} is {result}'.format(split=split_num,
            #                                                                 result=result))
    elif should_load:
        image_data = get_image_train_data(DEFAULT_SPECTROGRAM_PATH)
        model = tf.keras.models.load_model(EXPORT_PATH)
    else:
        raise RuntimeError('Unreachable flow reached')

    image_batch, label_batch = image_data[0]

    class_names = sorted(image_data.class_indices.items(), key=lambda pair: pair[1])
    class_names = np.array([key.title() for key, value in class_names])
    print(class_names)
    predicted_batch = model.predict(image_batch)
    predicted_id = np.argmax(predicted_batch, axis=-1)
    predicted_label_batch = class_names[predicted_id]

    label_id = np.argmax(label_batch, axis=-1)
    plt.figure(figsize=(10, 9))
    plt.subplots_adjust(hspace=0.5)
    for n in range(min(30, configuration.batch_size)):
        plt.subplot(6, 5, n + 1)
        plt.imshow(image_batch[n])
        color = 'green' if predicted_id[n] == label_id[n] else 'red'
        plt.title(predicted_label_batch[n].title(), color=color)
        plt.axis('off')
    _ = plt.suptitle('Model predictions (green: correct, red: incorrect)')
    plt.show()

    # How to predict for 1 picture
    # image = tf.keras.preprocessing.image.load_img('7061-6-0-0.png', target_size=IMAGE_SHAPE)
    # np_image = np.array(image, dtype=float)
    # np_image = np.expand_dims(np_image, axis=0)
    # result = model.predict(np_image, batch_size=1)
