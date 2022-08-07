# Similar to testCNN.py, only the accuracy and confusion matrix are computed with all 3 categories
# besides diarrhea as one category

import numpy
from keras.models import load_model
from keras import preprocessing
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os, shutil
import librosa
import soundfile as sf


data_dir = "images/easy_test"
model_file = "bestCNN4.h5"
type = "online"
(img_height, img_width) = (640, 480)
batch_size = 8

input_shape = (img_width, img_height, 3)

datagen = preprocessing.image.ImageDataGenerator(
    rescale=1./255)

generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical')

model = load_model(model_file)

prediction = model.predict(generator)
predicted_class_indices = numpy.argmax(prediction, axis=1)

labels = generator.class_indices
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
filenames = generator.filenames
actual = []
for n in range(len(filenames)):
    slash_index = filenames[n].find("/")
    actual += [filenames[n][:slash_index]]

# create new arrays to merge non-diarrhea categories
binary_predictions = []
binary_actual = []
for n in range(len(filenames)):
    if predictions[n] == "diarrhea":
        binary_predictions += ["diarrhea"]
    else:
        binary_predictions += ["non-diarrhea"]

    if actual[n] == "diarrhea":
        binary_actual += ["diarrhea"]
    else:
        binary_actual += ["non-diarrhea"]

cf_matrix = confusion_matrix(binary_actual, binary_predictions)
accurate = cf_matrix[0][0] + cf_matrix[1][1]

for file in os.scandir("error_images"):
    os.remove(file.path)
for file in os.scandir("error_audio"):
    os.remove(file.path)

print("\nSorting misclassifications...")

print("\nMisclassified files:")
for n in range(len(filenames)):
    if (actual[n] != predictions[n]):
        print("Thought " + filenames[n] + " was " + predictions[n])

word = str(input("\nWould you like to store the misclassified audio/image samples? Type [y] for yes: "))
if word == 'y':
    for n in range(len(filenames)):
        if (actual[n] != predictions[n]):
            shutil.copy(data_dir + "/" + filenames[n], "error_images")

            slash_index = filenames[n].find("/")
            old_name = filenames[n][slash_index + 1 :]
            new_name = "File " + str(n) + "_actual " + actual[n] + ", predicted " + predictions[n]
            shutil.move("error_images/" + old_name, "error_images/" + new_name + ".png")

            mp3_index = filenames[n].find(".mp3")
            file_name = filenames[n][: mp3_index + 4]
            file_info = filenames[n][mp3_index + 5 :]

            dash = file_info.find("-")
            sec = file_info.find("sec_spec")
            first = int(file_info[: dash])
            second = int(file_info[dash + 1 : sec])

            samples, sample_rate = librosa.load("audio/" + type + "/" + file_name, sr=None, offset=first, duration=(second-first))
            sf.write("error_audio/" + new_name + ".wav", samples, samplerate=sample_rate)
    print("Stored misclassifications in error_images and _audio")

ax = sns.heatmap(cf_matrix, annot=True, cmap='Reds')

ax.set_title('Confusion Matrix (%.2f%% Total Accuracy)\n' % (100 * accurate / len(binary_predictions)))
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ')

ax.xaxis.set_ticklabels(["diarrhea", "non-diarrhea"])
ax.yaxis.set_ticklabels(["diarrhea", "non-diarrhea"])

plt.savefig("binary_results.png")
print("\nSaved accuracy and confusion matrix in binary_results.png")
