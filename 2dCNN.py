# Program to train and save a CNN classification model using the images in train, validate, and test
# Generates or replaces an h5 file with the input name

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten
from keras.layers import Dropout, Dense
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import Sequential
from keras import preprocessing
from keras import regularizers
import wandb
from wandb.keras import WandbCallback
import matplotlib.pyplot as plt


# file to save model weights and structure to
model_file = 'bestCNN14.h5'
# hyper-parameter configuration for Weights and Biases
config={
    "batch_size": 4, # number of images processed during training before weights update
    "epochs": 1, # number of times the model passes through the entire training dataset
    "dropout_c": 0.3, # dropout from convolutional layers
    "dropout_ga": 0.1, # dropout from global averaging layer
    "num_features_c1": 64, # ""
    "num_features_c2": 64, # sizes of feature maps in respective convolutional blocks
    "num_features_c3": 128, # ""
    "num_features_c4": 64, # ""
    "learning_rate": 0.0003
}
wandb.init(project="Feces Thesis Anthony", entity="apopa6", config=config)

(img_height, img_width) = (640, 480)
input_shape = (img_width, img_height, 3)


# data is split into three categories:
# training for tuning the network
# validation to check performance during training
# testing to check performance after training
train_dir = "images/train"
val_dir = "images/validate"
test_dir = "images/test"

# process data from the sub-directories into categories (diarrhea, farts, pee, and poop)
datagen = preprocessing.image.ImageDataGenerator(
    rescale=1./255)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=wandb.config.batch_size,
    class_mode='categorical')

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=wandb.config.batch_size,
    class_mode='categorical')

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=wandb.config.batch_size,
    class_mode='categorical')

# model consists of 4 convolutional blocks, each following a similar structure:
# 1. convolutional layer
# 2. max pooling layer
# 3. dropout
# the activations of the final convolutional block are pooled together in the global average pooling layer.
# the network then reduces to 4 final neurons, providing probability outputs for each category.
model = Sequential()

model.add(Conv2D(wandb.config.num_features_c1, (3, 3), input_shape=input_shape, activation="relu", padding="valid"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(wandb.config.dropout_c))
 
model.add(Conv2D(wandb.config.num_features_c2, (3, 3), activation="relu", padding="valid"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(wandb.config.dropout_c))
 
model.add(Conv2D(wandb.config.num_features_c3, (3, 3), activation="relu", padding="valid"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(wandb.config.dropout_c))

model.add(Conv2D(wandb.config.num_features_c4, (3, 3), activation="relu", padding="valid"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(wandb.config.dropout_c))

model.add(GlobalAveragePooling2D())
model.add(Dropout(wandb.config.dropout_ga))
model.add(Dense(4, activation="softmax"))

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=wandb.config.learning_rate),
              metrics=['accuracy'])

# the model during training with the lowest validation loss is saved as the final model
modelCheckpoint = ModelCheckpoint(model_file, monitor='val_loss', mode='min', save_best_only=True)
history = model.fit(train_generator, validation_data=val_generator, epochs=wandb.config.epochs, batch_size=wandb.config.batch_size, 
            callbacks=[WandbCallback(), modelCheckpoint])

# save model history for later use
with open('/trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
    # to load use: history = pickle.load(open('/trainHistoryDict', "rb")

# create plots of loss on training and validation data
print("Creating plots")
plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.title('Loss vs. Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['validation loss', 'training loss'], loc='upper left')
plt.savefig("loss.eps", format='eps')
plt.savefit('loss.png')

# the saved model is loaded and evaluated on the testing data
saved_model = load_model(model_file)
scores = saved_model.evaluate(test_generator)
test_loss = round(scores[0], 4)
test_accuracy = round(100 * scores[1], 2)
wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy, "config": config})
