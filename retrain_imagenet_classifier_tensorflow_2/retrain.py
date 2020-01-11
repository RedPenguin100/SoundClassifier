import matplotlib.pylab as plt
import tensorflow as tf

import tensorflow_hub as hub
from tensorflow.keras import layers
import time
import numpy as np
import sys
import PIL.Image as Image

from retrain_imagenet_classifier_tensorflow_2.sound_to_image import wav_to_spectogram
def fix_gpu():
    """
    I don't fully understand this yet but these lines fix an error in CUDNN
    """

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))



class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        self.model.reset_metrics()

DATASET_SIZE = 8732
IMAGE_SHAPE = (224, 224)
DATA_ROOT = r'spectrograms'
EXPORT_PATH = 'C:\\tmp\\saved_models\\spectogram'
FEATURE_EXTRACTOR_URL = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2'

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError("Usage: {} load/predict".format(sys.argv[0]))
    should_load = sys.argv[1] == 'load'
    fix_gpu()

    # Obtain data to memory.
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
    image_data = image_generator.flow_from_directory(str(DATA_ROOT), target_size=IMAGE_SHAPE, batch_size=12)
    image_batch, label_batch = image_data[0]

    # Now we want to retrain:
    feature_extractor_layer = hub.KerasLayer(FEATURE_EXTRACTOR_URL,
                                             input_shape=(224, 224, 3))

    feature_extractor_layer.trainable = False

    # Add the final layer to the ready model to train.
    model = tf.keras.Sequential([feature_extractor_layer, layers.Dense(image_data.num_classes,
                                                                       activation='softmax')])
    print(model.summary())
    if not should_load:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='categorical_crossentropy',
            metrics=['acc'])

        steps_per_epoch = np.ceil(image_data.samples / image_data.batch_size)

        batch_stats_callback = CollectBatchStats()

        model.fit_generator(image_data, epochs=2,
                            steps_per_epoch=steps_per_epoch,
                            callbacks=[batch_stats_callback])
        model.save(EXPORT_PATH, save_format='tf')
    else:
        model = tf.keras.models.load_model(EXPORT_PATH)

    class_names = sorted(image_data.class_indices.items(), key=lambda pair: pair[1])
    class_names = np.array([key.title() for key, value in class_names])
    print(class_names)
    predicted_batch = model.predict(image_batch)
    predicted_id = np.argmax(predicted_batch, axis=-1)
    predicted_label_batch = class_names[predicted_id]

    label_id = np.argmax(label_batch, axis=-1)
    plt.figure(figsize=(10, 9))
    plt.subplots_adjust(hspace=0.5)
    for n in range(30):
        plt.subplot(6, 5, n + 1)
        plt.imshow(image_batch[n])
        color = 'green' if predicted_id[n] == label_id[n] else 'red'
        plt.title(predicted_label_batch[n].title(), color=color)
        plt.axis('off')
    _ = plt.suptitle('Model predictions (green: correct, red: incorrect)')
    plt.show()

    image = tf.keras.preprocessing.image.load_img('7061-6-0-0.png', target_size=IMAGE_SHAPE)
    np_image = np.array(image, dtype=float)
    np_image = np.expand_dims(np_image, axis=0)
    result = model.predict(np_image, batch_size=1)
    print(EXPORT_PATH)
