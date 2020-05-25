import pytest
import tensorflow as tf
from tensorflow.keras.models import load_model

from retrain_imagenet_classifier_tensorflow_2.retrain import EXPORT_PATH, get_image_test_data, fix_gpu, \
    get_all_data
from retrain_imagenet_classifier_tensorflow_2.sound_to_image import SPECTROGRAM_PATH


@pytest.fixture(scope='function', autouse=True)
def setup():
    fix_gpu()


def test_accuracy_10fold():
    avg_accuracy = 0
    avg_loss = 0
    total_images = 8732
    for split_num in range(1, 11):
        model = load_model(EXPORT_PATH + str(split_num) + 'split')
        test_data = get_image_test_data(SPECTROGRAM_PATH, split_num)
        weight = test_data.samples
        result = model.evaluate_generator(test_data)
        print(result)
        avg_loss += result[0] * weight / total_images
        avg_accuracy += result[1] * weight / total_images
    print("Average accuracy for the model: {}".format(avg_accuracy))
    print("Average loss for the model: {}".format(avg_loss))


def test_accuracy_naive():
    model = load_model(EXPORT_PATH)
    test_data = get_all_data()
    result = model.evaluate_generator(test_data)
    print(result)
