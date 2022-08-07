# Program for creating and analyzing predictions from a CNN on images from a given directory

import numpy
from keras.models import load_model
from keras import preprocessing
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os, shutil
import librosa
import soundfile as sf


# file from which to load model weights and structure
test_data = 'sim' # data to predict on (easy_test = no background noise, test = with background noise, sim = machine noise)
data_dir = 'images/' + test_data
(img_height, img_width) = (640, 480)
batch_size = 8

# 3 channels means images are input in RGB
input_shape = (img_width, img_height, 3)

# process data from the directory into categories (diarrhea, farts, pee, and poop)
datagen = preprocessing.image.ImageDataGenerator(
    rescale=1./255)

generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical')

model = load_model('bestCNN11.h5')

# for each image from the generator, 4 probabilities are output for each category that image could be in
prediction = model.predict(generator)
# the highest probability is used as the prediction
predicted_class_indices = numpy.argmax(prediction, axis=1)

# dictionary maps category names to indices, gets reversed and used to create an array of predicted names
labels = generator.class_indices
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
filenames = generator.filenames

# then, an array of the actual names is created
actual = []
for n in range(len(filenames)):
    slash_index = filenames[n].find("/")
    actual += [filenames[n][:slash_index]]

# generate confusion matrix and compute number of accurate classifications by summing along diagonal
cf_matrix = confusion_matrix(actual, predictions)
accurate = 0
for n in range(len(labels)):
    accurate += cf_matrix[n][n]

# clear error images and audio directories
for file in os.scandir("error_images"):
    os.remove(file.path)
for file in os.scandir("error_audio"):
    os.remove(file.path)

print("\nSorting misclassifications...")

# display names of misclassified files in terminal
print("\nMisclassified files:")
for n in range(len(filenames)):
    # if the names do not match, a misclassification occurred
    if (actual[n] != predictions[n]):
        print("Thought " + filenames[n] + " was " + predictions[n])

# give the user the option to populate error images and audio directories with misclassified spectrograms and audio samples
word = str(input("\nWould you like to store the misclassified audio/image samples? Type [y] for yes: "))
if word == 'y':
    for n in range(len(filenames)):
        if (actual[n] != predictions[n]):
            # copy the misclassified spectrograms to error_images
            shutil.copy(data_dir + "/" + filenames[n], "error_images")

            # parse the filename to extract useful information about the audio
            mp3_index = filenames[n].find(".mp3")
            type = "online"
            if mp3_index == -1:
                mp3_index = filenames[n].find(".wav")
                type = "simulated"
            file_name = filenames[n][: mp3_index + 4]
            file_info = filenames[n][mp3_index + 5 :]

            # rename the spectrograms in error_images so that they state the real name vs predicted name
            slash_index = filenames[n].find("/")
            old_name = filenames[n][slash_index + 1 :]
            new_name = filenames[n][slash_index + 1 : mp3_index] + ", predicted " + predictions[n]
            shutil.move("error_images/" + old_name, "error_images/" + new_name + ".png")

            # more parsing to figure out which seconds from the audio are targeted
            dash = file_info.find("-")
            sec = file_info.find("sec_spec")
            first = int(file_info[: dash])
            second = int(file_info[dash + 1 : sec])

            # having worked backwards to get the information about the audio sample, load and save the sample as a .wav file
            samples, sample_rate = librosa.load("audio/" + type + "/" + file_name, sr=None, offset=first, duration=(second-first))
            sf.write("error_audio/" + new_name + ".wav", samples, samplerate=sample_rate)
    print("Stored misclassifications in error_images and _audio")

# setup for confusion matrix display
ax = sns.heatmap(cf_matrix, annot=True, cmap='Reds', fmt='g', cbar=False)

ax.set_title('Confusion Matrix (%.2f%% Total Accuracy)\n' % (100 * accurate / len(predictions)),fontsize=14)
ax.set_xlabel('Predicted Values', fontsize=14)
ax.set_ylabel('Actual Values ', fontsize=14)

ax.xaxis.set_ticklabels(["diarrhea", "flatulence", "urination", "defecation"], fontsize=12)
ax.yaxis.set_ticklabels(["diarrhea", "flatulence", "urination", "defecation"], fontsize=12)

# save confusion matrix with percent accuracy in title
plt.savefig(test_data+'-results.eps', format='eps')
print("\nSaved accuracy and confusion matrix in results")
