# coding: utf-8

import csv
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

count = -1
URBAN_SOUND8K_PATH = os.path.join(os.path.pardir, 'UrbanSound8K')
URBAN_SOUND8K_CSV_PATH = os.path.join(URBAN_SOUND8K_PATH, os.path.join('metadata', 'UrbanSound8K.csv'))
AUDIO_PATH = os.path.join(URBAN_SOUND8K_PATH, 'audio')
SPECTROGRAM_PATH = 'spectrogram-v2.0'


def wav_to_spectogram(wav_path, spectogram_path):
    frames, rate = librosa.load(wav_path)
    S = librosa.feature.melspectrogram(frames, sr=rate, n_mels=128)

    log_S = librosa.amplitude_to_db(S, ref=np.max)

    # Make a new figure
    fig = plt.figure(figsize=(12, 4))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Display the spectrogram on a mel scale
    # sample rate and hop length parameters are used to render the time axis
    librosa.display.specshow(log_S, sr=rate, x_axis='time', y_axis='mel')

    # Make the figure layout compact
    plt.savefig(spectogram_path)
    plt.close()


if __name__ == '__main__':
    with open(URBAN_SOUND8K_CSV_PATH) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            count += 1

            if count == 0:
                continue

            wavfile_name = str(row[0])
            fold = str(row[5])
            label = str(row[7])
            wavpath = os.path.join(AUDIO_PATH, 'fold' + fold) + os.path.sep + wavfile_name

            spectrogram_path = os.path.join(SPECTROGRAM_PATH, label, wavfile_name + '.png')
            os.makedirs(os.path.dirname(spectrogram_path), exist_ok=True)
            wav_to_spectogram(wavpath, spectrogram_path)
            print(count)
