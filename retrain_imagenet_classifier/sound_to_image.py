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


with open(URBAN_SOUND8K_CSV_PATH) as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        count += 1

        if count == 0:
            continue

        print(count)
        if not os.path.exists('spectrograms/' + row[7]):
            os.makedirs('spectrograms/' + row[7])

        y, sr = librosa.load(os.path.join(AUDIO_PATH, 'fold') + str(row[5]) + os.path.sep + str(row[0]))

        # Let's make and display a mel-scaled power (energy-squared) spectrogram
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

        # Convert to log scale (dB). We'll use the peak power as reference.
        log_S = librosa.amplitude_to_db(S, ref=np.max)

        # Make a new figure
        fig = plt.figure(figsize=(12, 4))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # Display the spectrogram on a mel scale
        # sample rate and hop length parameters are used to render the time axis
        librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

        # Make the figure layout compact

        # plt.show()
        plt.savefig('spectrograms/' + row[7] + '/' + row[0] + '.png')
        plt.close()

        print(count)
