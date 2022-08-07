# Script to create all spectrograms in the given list of categories, based on information stored
# in the spectrograms of the originals image directory
# See full_spectros.py for comments detailing augmentation and librosa functionality (much of it is repeated here)

import librosa
import librosa.display
import matplotlib.pyplot as plt
from audiomentations import Compose, AddGaussianNoise, PitchShift, Shift, TimeMask, BandStopFilter, AddBackgroundNoise
import random
import numpy
import soundfile as sf
import os, shutil


categories = ["poop"]
location = "validate"
augment = True

# function to return random directory (from train, validate, and test) unless otherwise specified
def directory(location):
    if location == "random":
        rand = random.uniform(0, 100)
        dir = ""
        if rand < 50:
            dir = "train"
        elif rand < 75:
            dir = "validate"
        else:
            dir = "test"
        return dir
    else:
        return location

# function to create unaugmented mel-spectrogram from an audio sample
def create_original(header, label, fileName, short, sample_rate):
    img = "_spec0.png"
    print("Generating " + fileName + label + img)
    sgram = librosa.stft(short)
    sgram_mag, _ = librosa.magphase(sgram)
    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
    mel_sgram = librosa.amplitude_to_db(mel_scale_sgram)

    librosa.display.specshow(mel_sgram, sr=sample_rate)
    plt.savefig(header + label + img)

# function to create an augmented spectrogram for each entry in the train_augmentations array
def create_train_augmentations(header, label, fileName, short, sample_rate):
    for n in range(len(train_augmentations)):
        print("Creating augmentation " + str(n+1))
        augmented_samples = train_augmentations[n](short, sample_rate)
        
        img = "_spec" + str(n+1) + ".png"
        print("Generating " + fileName + label + img)
        sgram = librosa.stft(augmented_samples)
        sgram_mag, _ = librosa.magphase(sgram)
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
        mel_sgram = librosa.amplitude_to_db(mel_scale_sgram)

        librosa.display.specshow(mel_sgram, sr=sample_rate)
        plt.savefig(header + label + img)

# function to create spectrograms augmented only with background noise for test/validate
def create_testval_augmentations(header, label, fileName, short, sample_rate):
    for n in range(1, testval_num):
        print("Creating augmentation " + str(n))
        augmented_samples = background_noise(short, sample_rate=sample_rate)
        
        img = "_spec" + str(n) + ".png"
        print("Generating " + fileName + label + img)
        sgram = librosa.stft(augmented_samples)
        sgram_mag, _ = librosa.magphase(sgram)
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
        mel_sgram = librosa.amplitude_to_db(mel_scale_sgram)

        librosa.display.specshow(mel_sgram, sr=sample_rate)
        plt.savefig(header + label + img)


# list of individual audio augmentations
shift = Shift(p=1)    # rotate shift the audio within the 10-sec sample by a random amount
pitch_shift = PitchShift(p=1)    # shift the pitch of the audio by a random amount of notes
gaussian_noise = AddGaussianNoise(min_amplitude=0.003, max_amplitude=0.005, p=1)    # add uniform static
# add environmental background noise, chosen randomly from the ESC-50 dataset (stored locally)
background_noise = AddBackgroundNoise("C:\\Users\\apopa6\\Desktop\\ESC-50\\audio", min_snr_in_db=5, max_snr_in_db=15, p=1)   
frequency_mask = BandStopFilter(p=1)  # block out a random band of frequencies
time_mask = TimeMask(p=1)  # block out a random time period, unless the filename starts with "audio_defecation"

# combined various augmentations above into sets of two and three
mask = Compose([frequency_mask, time_mask])
shift_bg_noise = Compose([shift, background_noise])
g_noise_freq_mask = Compose([gaussian_noise, frequency_mask])
g_noise_time_mask = Compose([gaussian_noise, time_mask])

# array storing all combined augmentations
train_augmentations = [shift, pitch_shift, mask, background_noise, shift_bg_noise, g_noise_freq_mask, g_noise_time_mask]
# total number of samples, original and augmented, generated for test or validate directories (provided augment = True)
testval_num = 3


fileName = "pee1min.mp3"
samples, sample_rate = librosa.load("audio\\online\\pee\\" + fileName, sr=None)

for category in categories:
    for file in os.listdir("images\\originals\\" + category):
        type = "online"
        mp3_index = file.find(".mp3")
        if mp3_index == -1:
            mp3_index = file.find(".wav")
            type = "simulated"

        name = file[: mp3_index + 4]
        if name != fileName:
            samples, sample_rate = librosa.load("audio\\" + type + "\\" + category + "\\" + name, sr=None)
            fileName = name

        rest = file[mp3_index + 5 :]
        dash_index = rest.find("-")
        sec_index = rest.find("sec_spec0")

        start_secs = int(rest[: dash_index])
        end_secs = int(rest[dash_index + 1 : sec_index])

        dir = directory(location)
        header = "images\\" + dir + "\\" + category + "\\" + fileName
        print("\nWriting to " + dir + " directory")
        label = "_" + str(start_secs) + "-" + str(end_secs) + "sec"

        # look at subsection of sample
        short = samples[start_secs * sample_rate : end_secs * sample_rate]
        # if the sample is shorter than 10 seconds, fill the remaining time with silence
        if (end_secs - start_secs < 10):
            difference = (10 - end_secs + start_secs) * sample_rate
            short = numpy.concatenate((short, numpy.zeros(difference)))

        # create the un-augmented and augmented spectrograms
        create_original(header, label, fileName, short, sample_rate)
        if augment:
            if dir == "train":
                create_train_augmentations(header, label, fileName, short, sample_rate)
            else:
                create_testval_augmentations(header, label, fileName, short, sample_rate)
        plt.cla()


for category in categories:
        for file in os.listdir("images\\validate\\" + category):
            name = file
            png_index = name.find(".png")
            name = name[:png_index]

            num = int(name[-1])
            if (num == 0):
                shutil.copy("images\\validate\\" + category + "\\" + file, "images\\easy_val\\" + category + "\\" + file)
        for file in os.listdir("images\\test\\" + category):
            name = file
            png_index = name.find(".png")
            name = name[:png_index]

            num = int(name[-1])
            if (num == 0):
                shutil.copy("images\\test\\" + category + "\\" + file, "images\\easy_test\\" + category + "\\" + file)
