# Script used to create spectrograms out of audio files in 'full_david_meyer'
# Similar to full_spectros.py, with no augmentations

import librosa
import librosa.display
import matplotlib.pyplot as plt
import random
import numpy
import soundfile as sf
import os


categories = ["diarrhea"]
location = "david_meyer_test"


def create_spec(header, fileName, label, short, sample_rate):
    img = "_spec0.png"
    print("Generating " + fileName + label + img)
    sgram = librosa.stft(short)
    sgram_mag, _ = librosa.magphase(sgram)
    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
    mel_sgram = librosa.amplitude_to_db(mel_scale_sgram)

    librosa.display.specshow(mel_sgram, sr=sample_rate)
    plt.savefig(header + label + img)


for category in categories:
    for fileName in os.listdir("audio\\full_david_meyer\\" + category):
        samples, sample_rate = librosa.load("audio\\full_david_meyer\\" + category + "\\" + fileName, sr=None)
        start_secs = 0
        end_secs = int(len(samples) / sample_rate)

        header = "images\\" + location + "\\" + category + "\\" + fileName
        print("\nWriting to " + location + " directory")

        if (end_secs - start_secs > 10):
            end_secs = start_secs + 10

        label = "_" + str(start_secs) + "-" + str(end_secs) + "sec"

        short = samples[start_secs * sample_rate : end_secs * sample_rate]
        if (end_secs - start_secs < 10):
            difference = (10 - end_secs + start_secs) * sample_rate
            short = numpy.concatenate((short, numpy.zeros(difference)))

        create_spec(header, fileName, label, short, sample_rate)
        plt.cla()
