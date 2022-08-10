# Script to generate the spectrogram figure comparison for journal
# A mel spectrogram logarithmically renders frequencies above a certain threshold (the corner frequency)

from tkinter import font
import librosa
import librosa.display
import matplotlib.pyplot as plt

category = "poop"    # one of the four categories for online audio
fileName = "327_0010_0_Jul-14-2022_11-31-49.wav"
split = True    # split into multiple 10-second intervals if true, treat audio as one sample if false
start_secs = 0    # number of seconds when to look
end_secs = 10     # if split = False, sound will be forced into a 10-second interval
if fileName[-4:] == '.mp3':
    location = "online"    
else:
    location = "simulated"
samples, sample_rate = librosa.load("audio/" + location + '/' + category + "/" + fileName, sr=None)

if category == 'diarrhea':
    title = 'Diarrhea'
elif category == 'poop':
    title = 'Defecation'
elif category == 'pee':
    title = 'Urination'
elif category == 'farts':
    title = 'Flatulence'

# function to create unaugmented mel-spectrogram from an audio sample
def create_original(header, label, fileName, short, sample_rate):
    img = "_spec0.png"
    print("Generating " + fileName + label + img)
    sgram = librosa.stft(short)
    sgram_mag, _ = librosa.magphase(sgram)
    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
    mel_sgram = librosa.amplitude_to_db(mel_scale_sgram)
    # plot spectrogram
    fig = plt.figure()
    if location == "online" and category == "poop":
        librosa.display.specshow(mel_sgram, sr=sample_rate, y_axis='mel')
        fig.gca().set(title=title, label='xlarge')
        fig.gca().set_ylabel("Mel-frequency (Hz)")
    elif location == "online" and category != "poop":
        librosa.display.specshow(mel_sgram, sr=sample_rate)
        fig.gca().set(title=title, label='xlarge')
    elif location == "simulated" and category == "poop":
        librosa.display.specshow(mel_sgram, sr=sample_rate, x_axis = 's', y_axis='mel')
        fig.gca().set_xticks(range(0, 11, 5))
        fig.gca().set_xlabel("Time (seconds)")
        fig.gca().set_ylabel("Mel-frequency (Hz)")
    elif location == "simulated" and category != "poop":
        librosa.display.specshow(mel_sgram, sr=sample_rate, x_axis = 's')
        fig.gca().set_xticks(range(0, 11, 5))
        fig.gca().set_xlabel("Time (seconds)")
    plt.savefig(header + label + img, dpi=600)


if split:
    # keep track of number of seconds into audio, use info to create label
    start = start_secs * sample_rate
    secs = start_secs
    label = "_" + str(secs) + "-" + str(secs + 10) + "sec"

    # while the end of the current sample is within the total audio length
    while (start + sample_rate * 10 <= len(samples)):
        header = "./specFig_imgs/" + location + "/" + category + "/" + fileName
        print("Writing to " + location + " directory")

        # look at current 10-sec interval of sample
        print("Starting sample at " + str(secs) + " seconds\n")
        short = samples[start: start + sample_rate * 10]
    
        # create the un-augmented and augmented spectrograms
        create_original(header, label, fileName, short, sample_rate)
        plt.cla()

        # increment seconds by 10 for next iteration and update label
        secs += 10
        if secs >= end_secs: # stop generating spectograms at a specific time
            break
        start += sample_rate * 10
        label = "_" + str(secs) + "-" + str(secs + 10) + "sec"
else: 
    print('nope')