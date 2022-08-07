# Program for experimenting with various strategies for generating spectrograms

import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, PitchShift, Shift, TimeMask, SpecFrequencyMask, AddBackgroundNoise, BandStopFilter
import numpy

samples, sample_rate = librosa.load("audio\\online\\diarrhea\\The Diarrhea Diaries.mp3", sr=None)

samples1 = samples[0: 10 * sample_rate]

img = "_spec0.png"
sgram = librosa.stft(samples1)
sgram_mag, _ = librosa.magphase(sgram)
mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
mel_sgram = librosa.amplitude_to_db(mel_scale_sgram)

librosa.display.specshow(mel_sgram, sr=sample_rate)

plt.colorbar(format='%+2.0f dB')
plt.savefig(img)

plt.clf()

samples, sample_rate = librosa.load("audio\\online\\pee\\female_pee.mp3", sr=None)

samples2 = samples[10 * sample_rate: 20 * sample_rate]

img = "_spec1.png"
sgram = librosa.stft(samples2)
sgram_mag, _ = librosa.magphase(sgram)
mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
mel_sgram = librosa.amplitude_to_db(mel_scale_sgram)

librosa.display.specshow(mel_sgram, sr=sample_rate)

plt.colorbar(format='%+2.0f dB')
plt.savefig(img)
