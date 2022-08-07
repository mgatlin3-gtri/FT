# Script to create spectrograms out of any file in the 'audio' directory

import librosa
import librosa.display
import matplotlib.pyplot as plt
from audiomentations import Compose, AddGaussianNoise, PitchShift, Shift, TimeMask, BandStopFilter, AddBackgroundNoise
import random
import numpy
import soundfile as sf


category = "poop"    # one of the four categories for online audio
fileName = "318_0010_0_Jul-14-2022_11-26-50.wav"
split = False    # split into multiple 10-second intervals if true, treat audio as one sample if false
start_secs = 0    # number of seconds when to look
end_secs = 10     # if split = False, sound will be forced into a 10-second interval
location = "originals"    # if "random", 50-25-25 probability split for train-validate-test
augment = False    # augmented versions of the data will be added if true 


# list of individual audio augmentations
time_shift = Shift(p=1)    # rotate shift the audio within the 10-sec sample by a random amount
gaussian_noise = AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.001, p=1)    # add uniform static
# add environmental background noise, chosen randomly from the ESC-50 dataset (stored locally)
background_noise = AddBackgroundNoise("C:\\Users\\apopa6\\Desktop\\ESC-50\\audio", min_snr_in_db=5, max_snr_in_db=15, p=1)   
frequency_mask = BandStopFilter(p=1)  # block out a random band of frequencies
# block out a random time period, unless the filename starts with "audio_defecation"
p_time_mask = 1
if fileName.find("audio_defecation") == 0:
    p_time_mask = 0
time_mask = TimeMask(p=p_time_mask)

# combined various augmentations above into sets of two and three
shift_g_noise = Compose([time_shift, gaussian_noise])
shift_bg_noise = Compose([time_shift, background_noise])
shift_mask = Compose([time_shift, frequency_mask, time_mask])
shift_mask_g_noise = Compose([time_shift, frequency_mask, time_mask, gaussian_noise])
shift_mask_bg_noise = Compose([time_shift, frequency_mask, time_mask, background_noise])

# array storing all combined augmentations
train_augmentations = [shift_g_noise, shift_bg_noise, shift_mask, shift_mask_g_noise, shift_mask_bg_noise]
# total number of samples, original and augmented, generated for test or validate directories (provided augment = True)
testval_num = 3

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
def create_original(header, label, short, sample_rate):
    img = "_spec0.png"
    print("Generating " + fileName + label + img)
    sgram = librosa.stft(short)
    sgram_mag, _ = librosa.magphase(sgram)
    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
    mel_sgram = librosa.amplitude_to_db(mel_scale_sgram)

    librosa.display.specshow(mel_sgram, sr=sample_rate)
    plt.savefig(header + label + img)


# function to create an augmented spectrogram for each entry in the train_augmentations array
def create_train_augmentations(header, label, short, sample_rate):
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
def create_testval_augmentations(header, label, short, sample_rate):
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


samples, sample_rate = librosa.load("audio\\simulated\\" + category + "\\" + fileName, sr=None)

if split:

    # keep track of number of seconds into audio, use info to create label
    start = start_secs * sample_rate
    secs = start_secs
    label = "_" + str(secs) + "-" + str(secs + 10) + "sec"

    # while the end of the current sample is within the total audio length
    while (start + sample_rate * 10 <= len(samples)):
        # decide which directory to put images in, based on value of location 
        dir = directory(location)
        header = "images\\" + dir + "\\" + category + "\\" + fileName
        print("\nWriting to " + dir + " directory")

        # look at current 10-sec interval of sample
        print("Starting sample at " + str(secs) + " seconds\n")
        short = samples[start: start + sample_rate * 10]
    
        # create the un-augmented and augmented spectrograms
        create_original(header, label, short, sample_rate)
        if augment:
            if dir == "train":
                create_train_augmentations(header, label, short, sample_rate)
            else:
                create_testval_augmentations(header, label, short, sample_rate)
        plt.cla()

        # increment seconds by 10 for next iteration and update label
        secs += 10
        if secs >= end_secs:
            break
        start += sample_rate * 10
        label = "_" + str(secs) + "-" + str(secs + 10) + "sec"
            
else:
    # decide which directory to put images in, based on value of location 
    dir = directory(location)
    header = "images\\" + dir + "\\" + category + "\\" + fileName
    print("\nWriting to " + dir + " directory")

    # modify value of end_secs to fit in 10-sec window and to not go past the audio length
    if (end_secs - start_secs > 10):
        end_secs = start_secs + 10
    if end_secs * sample_rate > len(samples):
        end_secs = int(len(samples) / sample_rate)

    label = "_" + str(start_secs) + "-" + str(end_secs) + "sec"
    
    # look at subsection of sample
    short = samples[start_secs * sample_rate : end_secs * sample_rate]
    
    noise, noise_sr = librosa.load("audio\\simulated\\other\\207_0000_0_Jul-05-2022_15-21-49.wav", sr=None)

    if (end_secs - start_secs < 10):
        difference = (10 - end_secs + start_secs) * sample_rate
        noise = noise[: difference * noise_sr]
        short = numpy.concatenate((short, noise))

    # create the un-augmented and augmented spectrograms
    create_original(header, label, short, sample_rate)
    if augment:
        if dir == "train":
            create_train_augmentations(header, label, short, sample_rate)
        else:
            create_testval_augmentations(header, label, short, sample_rate)
    plt.cla()
